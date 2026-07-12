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
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Whisper decoder max target length (model rejects longer label sequences).
WHISPER_MAX_LABEL_LENGTH = 448
# Whisper is trained on 30-second chunks; longer clips slow training enormously.
WHISPER_MAX_AUDIO_SECONDS = 30.0

from src.config import DOMAINS, SAMPLE_RATE  # noqa: E402
from src.whisper_dataset import (  # noqa: E402
    COMPETITION_ANV_LANGUAGES,
    TrainingRecord,
    audio_skip_count,
    balance_records_by_language,
    collect_records,
    format_duration_summary,
    subsample_eval_records,
    summarize_record_durations,
    summarize_records,
    try_load_record_audio,
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
    truncated_label_count: int = 0
    capped_audio_count: int = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features: list[dict[str, Any]] = []
        label_features: list[dict[str, Any]] = []
        max_audio_samples = int(WHISPER_MAX_AUDIO_SECONDS * SAMPLE_RATE)

        for example in features:
            record = record_from_batch_row(example)
            labels = self.processor.tokenizer(record.sentence).input_ids
            if len(labels) > WHISPER_MAX_LABEL_LENGTH:
                self.truncated_label_count += 1
                if self.truncated_label_count <= 20 or self.truncated_label_count % 100 == 0:
                    print(
                        f"WARNING: truncating long transcript ({self.truncated_label_count} total) "
                        f"key={record.key} tokens={len(labels)} -> {WHISPER_MAX_LABEL_LENGTH}",
                        flush=True,
                    )
                labels = labels[:WHISPER_MAX_LABEL_LENGTH]

            audio = try_load_record_audio(record, sample_rate=SAMPLE_RATE)
            if audio is None:
                continue

            audio_array = audio["array"]
            if len(audio_array) > max_audio_samples:
                self.capped_audio_count += 1
                if self.capped_audio_count <= 20 or self.capped_audio_count % 100 == 0:
                    duration_sec = len(audio_array) / SAMPLE_RATE
                    print(
                        f"WARNING: capping long audio ({self.capped_audio_count} total) "
                        f"key={record.key} {duration_sec:.1f}s -> {WHISPER_MAX_AUDIO_SECONDS}s "
                        "(clip kept in training)",
                        flush=True,
                    )
                audio_array = audio_array[:max_audio_samples]

            feats = self.processor.feature_extractor(
                audio_array,
                sampling_rate=audio["sampling_rate"],
            ).input_features[0]
            input_features.append({"input_features": feats})
            label_features.append({"input_ids": labels})

        if not input_features:
            raise RuntimeError(
                f"Entire batch had invalid audio ({audio_skip_count()} skips so far). "
                "Check pipeline outputs for corrupt parquet rows."
            )

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        if labels.shape[1] > WHISPER_MAX_LABEL_LENGTH:
            labels = labels[:, :WHISPER_MAX_LABEL_LENGTH]

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
    parser.add_argument(
        "--balance-languages",
        choices=["none", "equal", "cap"],
        default="none",
        help="Rebalance train set across languages. 'equal' uses the smallest language count; "
        "'cap' uses --max-samples-per-language.",
    )
    parser.add_argument(
        "--max-samples-per-language",
        type=int,
        default=None,
        help="Max train clips per language when --balance-languages=cap.",
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
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Cap dev clips used during training-time eval (full dev still used at end). "
        "Balanced across languages when set.",
    )
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--report-per-language", action="store_true", default=True)
    parser.add_argument("--no-report-per-language", action="store_false", dest="report_per_language")
    parser.add_argument(
        "--eval-all-languages",
        action="store_true",
        help="After training, evaluate dev WER for every language (track catastrophic forgetting).",
    )
    parser.add_argument(
        "--report-to",
        nargs="+",
        default=["tensorboard"],
        choices=["wandb", "tensorboard", "none"],
        help="Logging backends (default: tensorboard). Use: --report-to wandb tensorboard",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "asr-competition"),
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME"),
        help="Weights & Biases run name (default: output folder name).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases team/user (optional; leave unset to use API key default).",
    )
    parser.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP"),
        help="Weights & Biases run group, e.g. smoke-test or swahili-full.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="Resume from a checkpoint path, or pass flag alone to use the latest checkpoint in --output-dir.",
    )
    return parser.parse_args()


def resolve_report_to(report_to: list[str]) -> list[str]:
    if "none" in report_to:
        return []
    return report_to


def configure_wandb_env(args: argparse.Namespace, output_dir: Path) -> None:
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    else:
        os.environ.pop("WANDB_ENTITY", None)
    if args.wandb_group:
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group
    run_name = args.wandb_run_name or output_dir.name
    os.environ["WANDB_NAME"] = run_name
    os.environ.setdefault("WANDB_LOG_MODEL", "false")


def split_report_to(report_to: list[str]) -> tuple[bool, list[str]]:
    """Keep wandb out of HuggingFace report_to; we log to wandb via SafeWandbCallback."""
    if "none" in report_to:
        return False, []
    want_wandb = "wandb" in report_to
    trainer_report_to = [backend for backend in report_to if backend != "wandb"]
    return want_wandb, trainer_report_to


def log_to_wandb(metrics: dict[str, float]) -> None:
    if not metrics:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics)
    except Exception as exc:
        print(f"WARNING: wandb log failed ({exc})", file=sys.stderr)


class SafeWandbCallback(TrainerCallback):
    """Optional wandb logging that never raises — training always continues."""

    def __init__(self, args: argparse.Namespace, output_dir: Path, run_config: dict[str, Any]) -> None:
        self.args = args
        self.output_dir = output_dir
        self.run_config = run_config
        self.active = False

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
        except ImportError:
            print("WARNING: wandb not installed; training continues without wandb.", file=sys.stderr)
            return
        configure_wandb_env(self.args, self.output_dir)
        try:
            run = wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.wandb_run_name or self.output_dir.name,
                group=self.args.wandb_group,
                config=self.run_config,
                mode="online",
            )
            self.active = run is not None
            if self.active and wandb.run is not None and getattr(wandb.run, "url", None):
                print(f"wandb: {wandb.run.url}")
        except Exception as exc:
            self.active = False
            print(f"WARNING: wandb init failed ({exc}); training continues without wandb.", file=sys.stderr)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.active or not logs:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(logs, step=state.global_step)
        except Exception as exc:
            print(f"WARNING: wandb log failed ({exc})", file=sys.stderr)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.active:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish(quiet=True)
        except Exception:
            pass


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
    if args.balance_languages != "none":
        before = summarize_records(train_records)
        train_records = balance_records_by_language(
            train_records,
            mode=args.balance_languages,
            max_per_language=args.max_samples_per_language,
            seed=args.seed,
        )
        print(f"Balanced train ({args.balance_languages}): {before} -> {summarize_records(train_records)}")
    eval_records = collect_records(split=args.eval_split, **common_kwargs)
    eval_records_for_training = subsample_eval_records(
        eval_records,
        max_samples=args.max_eval_samples,
        seed=args.seed,
    )

    print(f"Work dir: {work_dir}")
    print(f"Train samples: {len(train_records)} — {summarize_records(train_records)}")
    print(f"Eval samples:  {len(eval_records)} — {summarize_records(eval_records)}")
    if args.max_eval_samples and len(eval_records_for_training) < len(eval_records):
        print(
            f"Training-time eval samples: {len(eval_records_for_training)} "
            f"(capped from {len(eval_records)}) — {summarize_records(eval_records_for_training)}"
        )
    train_duration = summarize_record_durations(train_records)
    print("Train audio duration (from manifest metadata):")
    print(format_duration_summary(train_duration))

    if args.dry_run:
        return 0

    if not train_records:
        print("ERROR: no train records found.", file=sys.stderr)
        return 1

    want_wandb, trainer_report_to = split_report_to(resolve_report_to(args.report_to))

    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds = records_to_dataset(train_records)
    eval_ds = records_to_dataset(eval_records_for_training) if eval_records_for_training else None

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
    max_steps = args.max_steps if args.max_steps is not None else -1
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_strategy="steps",
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
        report_to=trainer_report_to,
        run_name=args.wandb_run_name or output_dir.name,
        push_to_hub=False,
    )

    run_config = {
        "model_name": args.model_name,
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "train_languages": summarize_records(train_records),
        "eval_languages": summarize_records(eval_records),
        "swahili_domains": args.swahili_domains,
        "anv_languages": args.anv_languages,
        "balance_languages": args.balance_languages,
        "max_samples_per_language": args.max_samples_per_language,
        "max_steps": max_steps,
        "num_train_epochs": args.num_train_epochs,
    }
    callbacks: list[TrainerCallback] = []
    if want_wandb:
        callbacks.append(SafeWandbCallback(args, output_dir, run_config))

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        processing_class=processor,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    skipped = audio_skip_count()
    if skipped:
        print(f"Training skipped {skipped} corrupt/unreadable audio clip(s).")
    if collator.truncated_label_count:
        print(
            f"Training truncated {collator.truncated_label_count} transcript(s) "
            f"to {WHISPER_MAX_LABEL_LENGTH} tokens (clips still used)."
        )
    if collator.capped_audio_count:
        print(
            f"Training capped {collator.capped_audio_count} clip(s) "
            f"to first {WHISPER_MAX_AUDIO_SECONDS}s (clips still used)."
        )

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
                log_to_wandb({f"wer/dev/{language}": wer for language, wer in per_lang.items()})
                log_to_wandb({"wer/dev/average": avg_wer})

        if args.eval_all_languages:
            all_eval_records = collect_records(
                work_dir=work_dir,
                split=args.eval_split,
                swahili_domains=tuple(DOMAINS),
                anv_languages=tuple(COMPETITION_ANV_LANGUAGES),
                include_swahili=True,
                include_anv=True,
                skip_maasai_scripted_train=not args.include_maasai_scripted_train,
                seed=args.seed,
            )
            if all_eval_records:
                print("\n=== All-language dev WER (sequential / forgetting check) ===")
                chain_wer = evaluate_per_language(trainer, all_eval_records, processor, metric)
                for language, wer in chain_wer.items():
                    print(f"  {language}: {wer:.2f}%")
                if chain_wer:
                    chain_avg = float(np.mean(list(chain_wer.values())))
                    print(f"Unweighted average WER (all langs): {chain_avg:.2f}%")
                    chain_path = output_dir / "all_language_wer.json"
                    chain_path.write_text(json.dumps(chain_wer, indent=2), encoding="utf-8")
                    log_to_wandb({f"wer/chain/{language}": wer for language, wer in chain_wer.items()})
                    log_to_wandb({"wer/chain/average": chain_avg})

    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    print(f"Saved model to {output_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
