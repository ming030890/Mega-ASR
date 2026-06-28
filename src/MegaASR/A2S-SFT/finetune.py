# coding=utf-8
import numpy as np
import torch
from transformers import EarlyStoppingCallback, TrainingArguments

from arguments import parse_args
from checkpointing import MakeCheckpointInferableCallback, find_latest_checkpoint
from dataloader import Qwen3ASRCollator, build_datasets
from modeling import apply_lora, load_qwen3_asr
from trainer import MegaASRTrainer


def build_training_args(args, use_bf16: bool):
    report_to = [] if args.report_to.lower() in ["", "none"] else [args.report_to]
    # PEFT/LoRA checkpoints contain adapter_model.safetensors, not a full
    # pytorch_model.bin. Transformers' best-model reload path expects full model
    # weights and fails at end of training, after a successful LoRA run.
    load_best_model_at_end = bool(
        args.eval_file
        and args.early_stopping_patience > 0
        and not bool(args.use_lora)
    )

    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=bool(args.pin_memory),
        dataloader_persistent_workers=bool(args.persistent_workers),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=bool(args.eval_file),
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=report_to,
        run_name="Mega-ASR-A2S-SFT",
    )


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def compute_target_metrics(eval_pred):
    predictions, labels = eval_pred
    pred_ids = np.asarray(predictions)
    label_ids = np.asarray(labels)

    if pred_ids.ndim == 3:
        pred_ids = pred_ids.argmax(axis=-1)

    pred_ids = pred_ids[:, :-1]
    label_ids = label_ids[:, 1:]
    target_mask = label_ids != -100
    target_tokens = int(target_mask.sum())
    if target_tokens == 0:
        return {
            "target_token_accuracy": 0.0,
            "target_exact_match": 0.0,
            "target_tokens_per_example": 0.0,
        }

    token_matches = (pred_ids == label_ids) & target_mask
    active_examples = target_mask.any(axis=1)
    exact_by_example = (token_matches | ~target_mask).all(axis=1)

    return {
        "target_token_accuracy": float(token_matches.sum() / target_tokens),
        "target_exact_match": float(exact_by_example[active_examples].mean()),
        "target_tokens_per_example": float(target_mask.sum(axis=1)[active_examples].mean()),
    }


def main():
    args = parse_args()

    model, processor, use_bf16 = load_qwen3_asr(args.model_path)

    if args.padding_side != "auto":
        processor.tokenizer.padding_side = args.padding_side
    print("padding_side =", processor.tokenizer.padding_side)

    model = apply_lora(model, args)

    dataset = build_datasets(args.train_file, args.eval_file)
    collator = Qwen3ASRCollator(processor=processor, sampling_rate=args.sr)
    training_args = build_training_args(args, use_bf16)

    callbacks = [MakeCheckpointInferableCallback(args.model_path)]
    if args.eval_file and args.early_stopping_patience > 0 and not bool(args.use_lora):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    elif args.eval_file and args.early_stopping_patience > 0 and bool(args.use_lora):
        print(
            "[early_stopping] disabled for LoRA because PEFT checkpoints do not "
            "support Transformers load_best_model_at_end full-model reload"
        )

    trainer = MegaASRTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        data_collator=collator,
        compute_metrics=compute_target_metrics if args.eval_file else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if args.eval_file else None,
        processing_class=processor,
        callbacks=callbacks,
        processor=processor,
        base_model_path=args.model_path,
        merged_from_lora_path=args.merge_lora_into_base_from.strip(),
        lr_encoder=args.lr_encoder,
        lr_aligner=args.lr_aligner,
        lr_llm=args.lr_llm,
        per_example_target_loss=bool(args.per_example_target_loss),
    )

    resume_from = args.resume_from.strip()
    if not resume_from and args.resume:
        resume_from = find_latest_checkpoint(args.output_dir) or ""

    if resume_from:
        print(f"[resume] {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
