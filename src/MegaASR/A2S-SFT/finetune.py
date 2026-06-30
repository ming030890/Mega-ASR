# coding=utf-8
import inspect
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainingArguments
from transformers.trainer_utils import SchedulerType

from arguments import parse_args
from checkpointing import MakeCheckpointInferableCallback, find_latest_checkpoint
from dataloader import Qwen3ASRCollator, build_datasets
from metrics import build_compute_target_metrics
from modeling import apply_lora, load_qwen3_asr
from trainer import MegaASRTrainer


def best_model_metric_name(eval_file: str) -> str:
    eval_files = [value.strip() for value in eval_file.split(",") if value.strip()]
    if len(eval_files) <= 1:
        return "eval_loss"
    primary_name = Path(eval_files[0]).name.removesuffix(".jsonl")
    return f"eval_{primary_name}_loss"


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

    scheduler_type = args.lr_scheduler_type
    supported_schedulers = {item.value for item in SchedulerType}
    if scheduler_type not in supported_schedulers:
        print(
            f"[scheduler] {scheduler_type!r} is not supported by this Transformers "
            "version; falling back to 'cosine'"
        )
        scheduler_type = "cosine"

    training_kwargs = {}
    if (
        scheduler_type == "cosine_with_min_lr"
        and args.lr_scheduler_min_lr_rate > 0
        and "lr_scheduler_kwargs" in inspect.signature(TrainingArguments.__init__).parameters
    ):
        training_kwargs["lr_scheduler_kwargs"] = {
            "min_lr_rate": args.lr_scheduler_min_lr_rate,
        }

    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=scheduler_type,
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
        metric_for_best_model=best_model_metric_name(args.eval_file),
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=report_to,
        run_name="Mega-ASR-A2S-SFT",
        **training_kwargs,
    )


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def validation_datasets(dataset):
    validation_keys = [key for key in dataset.keys() if key.startswith("validation")]
    if not validation_keys:
        return None
    if validation_keys == ["validation"]:
        return dataset["validation"]
    return {
        key.removeprefix("validation_"): dataset[key]
        for key in sorted(validation_keys)
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
        eval_dataset=validation_datasets(dataset),
        data_collator=collator,
        compute_metrics=build_compute_target_metrics(processor) if args.eval_file else None,
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
