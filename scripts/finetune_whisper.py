#!/usr/bin/env python3
"""
Fine-tune openai/whisper-small on Afrivoice Swahili + Anv-ke multilingual data.

Usage (smoke test on Orchard):
  export WORK_DIR=/project/community/rmwisene/pipeline_outputs
  python scripts/finetune_whisper.py \\
    --work-dir "$WORK_DIR" \\
    --output-dir "$WORK_DIR/whisper_runs/smoke_agriculture" \\
    --swahili-domains agriculture \\
    --no-anv \\
    --max-samples-per-source 1000 \\
    --max-steps 200 \\
    --eval-steps 50 \\
    --per-device-train-batch-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import DOMAINS, SAMPLE_RATE  # noqa: E402
from src.whisper_dataset import (  # noqa: E402
    COMPETITION_ANV_LANGUAGES,
    TrainingRecord,
    collect_records,
    load_record_audio,
    summarize_records,
)


def record_from_batch_row(example: dict[str, Any]) -> TrainingRecord:
    return TrainingRecord(
        key=example["key"],
        sentence=example["sentence"],
        language=example["language"],
        audio_path=example["audio_path"] or None,
        audio_source=json.loads(example["audio_source"]) if example["audio_source"] else None,
        source=example.get("source", ""),
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Load audio lazily per batch so full multilingual training does not cache every clip."""

    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features: list[dict[str, Any]] = []
        label_features: list[dict[str, Any]] = []

        for example in features:
            record = record_from_batch_row(example)
            audio = load_record_audio(record, sample_rate=SAMPLE_RATE)
            feats = self.processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
            ).input_features[0]
            labels = self.processor.tokenizer(record.sentence).input_ids
            input_features.append({"input_features": feats})
            label_features.append({"input_ids": labels})

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on competition ASR data.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("WORK_DIR", REPO_ROOT / "outputs")),
        help="Pipeline output root (processed/, cleaned/, features/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, logs, and metrics.",
    )
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument(
        "--swahili-domains",
        nargs="+",
        default=list(DOMAINS),
        help="Swahili domains to include (default: all five).",
    )
    parser.add_argument(
        "--anv-languages",
        nargs="+",
        default=list(COMPETITION_ANV_LANGUAGES),
        help="Anv languages to include.",
    )
    parser.add_argument("--no-swahili", action="store_true", help="Train on Anv languages only.")
    parser.add_argument("--no-anv", action="store_true", help="Train on Swahili only.")
    parser.add_argument(
        "--include-maasai-scripted-train",
        action="store_true",
        help="Include maasai/train/scripted (partial extract; off by default).",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total train samples.")
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=None,
        help="Cap samples per domain/style (good for balanced smoke tests).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print dataset stats and exit.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--report-per-language", action="store_true", default=True)
    parser.add_argument("--no-report-per-language", action="store_false", dest="report_per_language")
    return parser.parse_args()


def records_to_dataset(records: list[TrainingRecord]) -> Dataset:
    if not records:
        raise ValueError("No training records found. Check --work-dir and split paths.")
    return Dataset.from_list([record.to_row() for record in records])


def evaluate_per_language(
    trainer: Seq2SeqTrainer,
    eval_records: list[TrainingRecord],
    processor: WhisperProcessor,
    metric: Any,
) -> dict[str, float]:
    wer_by_language: dict[str, float] = {}
    languages = sorted({record.language for record in eval_records})
    for language in languages:
        subset = [record for record in eval_records if record.language == language]
        if not subset:
            continue
        eval_ds = records_to_dataset(subset)
        predictions = trainer.predict(eval_ds, metric_key_prefix=f"wer_{language}")
        pred_ids = predictions.predictions
        label_ids = predictions.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer_by_language[language] = 100 * metric.compute(predictions=pred_str, references=label_str)
    return wer_by_language


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "work_dir": work_dir,
        "swahili_domains": tuple(args.swahili_domains),
        "anv_languages": tuple(args.anv_languages),
        "include_swahili": not args.no_swahili,
        "include_anv": not args.no_anv,
        "skip_maasai_scripted_train": not args.include_maasai_scripted_train,
        "seed": args.seed,
    }

    train_records = collect_records(
        split=args.train_split,
        max_samples=args.max_samples,
        max_samples_per_source=args.max_samples_per_source,
        **common_kwargs,
    )
    eval_records = collect_records(split=args.eval_split, **common_kwargs)

    print(f"Work dir: {work_dir}")
    print(f"Train samples: {len(train_records)} — {summarize_records(train_records)}")
    print(f"Eval samples:  {len(eval_records)} — {summarize_records(eval_records)}")

    if args.dry_run:
        return 0

    if not train_records:
        print("ERROR: no train records found.", file=sys.stderr)
        return 1

    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds = records_to_dataset(train_records)
    eval_ds = records_to_dataset(eval_records) if eval_records else None

    metric = evaluate.load("wer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=compute_metrics if eval_ds is not None else None,
        processing_class=processor,
    )

    trainer.train()

    if eval_ds is not None:
        overall = trainer.evaluate()
        print(f"Overall dev WER: {overall.get('eval_wer', overall.get('wer')):.2f}%")

        if args.report_per_language:
            per_lang = evaluate_per_language(trainer, eval_records, processor, metric)
            print("Per-language dev WER:")
            for language, wer in per_lang.items():
                print(f"  {language}: {wer:.2f}%")
            if per_lang:
                avg_wer = float(np.mean(list(per_lang.values())))
                print(f"Unweighted average WER: {avg_wer:.2f}%")
                metrics_path = output_dir / "per_language_wer.json"
                metrics_path.write_text(json.dumps(per_lang, indent=2), encoding="utf-8")

    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    print(f"Saved model to {output_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
