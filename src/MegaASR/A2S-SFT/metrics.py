# coding=utf-8
import re

import numpy as np


TARGET_PREFIX_RE = re.compile(r"^\s*language\s+Cantonese\s*<asr_text>\s*")
MIXED_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._-]*|.", re.DOTALL)


def strip_target_prefix(text: str) -> str:
    return TARGET_PREFIX_RE.sub("", text).strip()


def mixed_cer_tokens(text: str) -> list[str]:
    tokens = []
    for token in MIXED_TOKEN_RE.findall(text):
        if token.isspace():
            continue
        tokens.append(token)
    return tokens


def edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def build_compute_target_metrics(processor):
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

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
                "target_tokens_per_example": 0.0,
                "target_cer": 0.0,
                "target_cer_edits": 0.0,
                "target_cer_tokens": 0.0,
            }

        token_matches = (pred_ids == label_ids) & target_mask
        active_examples = target_mask.any(axis=1)

        cer_edits = 0
        cer_tokens = 0
        for pred_row, label_row, mask_row in zip(pred_ids, label_ids, target_mask):
            if not mask_row.any():
                continue
            pred_target = np.where(mask_row, pred_row, pad_id)
            label_target = np.where(mask_row, label_row, pad_id)
            pred_text = strip_target_prefix(
                tokenizer.decode(pred_target.tolist(), skip_special_tokens=True)
            )
            label_text = strip_target_prefix(
                tokenizer.decode(label_target.tolist(), skip_special_tokens=True)
            )
            pred_tokens = mixed_cer_tokens(pred_text)
            label_tokens = mixed_cer_tokens(label_text)
            cer_edits += edit_distance(pred_tokens, label_tokens)
            cer_tokens += len(label_tokens)

        return {
            "target_token_accuracy": float(token_matches.sum() / target_tokens),
            "target_tokens_per_example": float(target_mask.sum(axis=1)[active_examples].mean()),
            "target_cer": float(cer_edits / cer_tokens) if cer_tokens else 0.0,
            "target_cer_edits": float(cer_edits),
            "target_cer_tokens": float(cer_tokens),
        }

    return compute_target_metrics
