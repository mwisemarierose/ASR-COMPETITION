#!/usr/bin/env python3
"""
Generate competition submission CSV/Parquet from a fine-tuned Whisper checkpoint.

Default test source is the Kaggle bundle (anv-test-data-nt): all six languages,
including Swahili, as Parquet under swa/kik/kln/luo/mas/som. Training Swahili
was Afrivoice tar.xz; do not use WORK_DIR processed/swahili for the 41,733-row upload.

Usage:
  python scripts/generate_submission.py \\
    --model-dir /project/.../checkpoint-12500 \\
    --output /project/.../submission_v1.csv \\
    --kaggle-test-root /project/.../datasets/anv-test-data-nt
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import DOMAINS, SAMPLE_RATE  # noqa: E402
from src.whisper_dataset import (  # noqa: E402
    COMPETITION_ANV_LANGUAGES,
    TrainingRecord,
    collect_kaggle_nt_test_records,
    collect_records,
    load_submission_template,
    set_forced_language_prompt,
    submission_language_code,
    summarize_records,
    try_load_record_audio,
)

MAX_AUDIO_SECONDS = 30.0
GENERATION_MAX_LENGTH = 225
# Kaggle rejects empty transcription cells as null values.
FAILED_AUDIO_PLACEHOLDER = "."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ASR competition submission file.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("WORK_DIR", REPO_ROOT / "outputs")),
        help="Pipeline output root (only used with --test-source work_dir).",
    )
    parser.add_argument(
        "--test-source",
        choices=("kaggle_nt", "work_dir"),
        default=os.environ.get("TEST_SOURCE", "kaggle_nt"),
        help="kaggle_nt = official 41,733-row competition test; work_dir = local manifests.",
    )
    parser.add_argument(
        "--kaggle-test-root",
        type=Path,
        default=Path(
            os.environ.get(
                "KAGGLE_TEST_ROOT",
                "/project/community/rmwisene/datasets/anv-test-data-nt",
            )
        ),
        help="Root of digitalumuganda/anv-test-data-nt download.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Fine-tuned checkpoint or final/ directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV or Parquet path.",
    )
    parser.add_argument(
        "--swahili-split",
        default="test",
        help="Manifest split for Swahili (under processed/{domain}/).",
    )
    parser.add_argument(
        "--anv-split",
        default="dev_test",
        help="Manifest split for Anv languages (under cleaned/{language}/).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--audio-workers",
        type=int,
        default=int(os.environ.get("AUDIO_WORKERS", "8")),
        help="Parallel CPU workers for parquet read + audio decode (main speed lever).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default=os.environ.get("INFER_DTYPE", "auto"),
        help="GPU inference dtype (auto=bf16 on CUDA).",
    )
    parser.add_argument("--max-audio-seconds", type=float, default=MAX_AUDIO_SECONDS)
    parser.add_argument(
        "--force-language-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force per-language decoder prompts at inference (epoch-2+ models).",
    )
    parser.add_argument("--id-column", default="id", help="Submission ID column name.")
    parser.add_argument(
        "--language-column",
        default="language",
        help="Submission language column name.",
    )
    parser.add_argument(
        "--text-column",
        default="transcription",
        help="Submission transcription column name.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=41733,
        help="Expected row count for competition upload (0 = skip check).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print test-set stats and exit.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap clips (smoke test).")
    return parser.parse_args()


def load_audio_for_record(record: TrainingRecord, max_audio_seconds: float) -> dict[str, Any] | None:
    audio = try_load_record_audio(record, sample_rate=SAMPLE_RATE)
    if audio is None:
        return None
    max_samples = int(max_audio_seconds * SAMPLE_RATE)
    if len(audio["array"]) > max_samples:
        audio = {
            "array": audio["array"][:max_samples],
            "sampling_rate": audio["sampling_rate"],
        }
    return audio


def resolve_autocast_dtype(device: torch.device, dtype: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if dtype == "fp32":
        return None
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    # auto: bf16 on Ampere+ (H100), else fp16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_batch_audio(
    batch_records: list[TrainingRecord],
    max_audio_seconds: float,
    audio_workers: int,
) -> list[tuple[TrainingRecord, dict[str, Any] | None]]:
    if audio_workers <= 1 or len(batch_records) <= 1:
        return [
            (record, load_audio_for_record(record, max_audio_seconds))
            for record in batch_records
        ]

    with ThreadPoolExecutor(max_workers=min(audio_workers, len(batch_records))) as pool:
        audios = pool.map(
            lambda record: load_audio_for_record(record, max_audio_seconds),
            batch_records,
        )
    return list(zip(batch_records, audios))


def transcribe_language_batch(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    records: list[TrainingRecord],
    *,
    batch_size: int,
    max_audio_seconds: float,
    audio_workers: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> list[tuple[str, str, str]]:
    outputs: list[tuple[str, str, str]] = []
    skipped_audio = 0
    language_code = submission_language_code(records[0].language) if records else "swa"

    for start in tqdm(range(0, len(records), batch_size), desc="  batches", leave=False):
        batch_records = records[start : start + batch_size]
        pending: list[tuple[TrainingRecord, dict[str, Any]]] = []

        for record, audio in load_batch_audio(batch_records, max_audio_seconds, audio_workers):
            if audio is None:
                skipped_audio += 1
                outputs.append((record.key, language_code, FAILED_AUDIO_PLACEHOLDER))
                continue
            pending.append((record, audio))

        if not pending:
            continue

        input_features = [
            {
                "input_features": processor.feature_extractor(
                    audio["array"],
                    sampling_rate=audio["sampling_rate"],
                ).input_features[0]
            }
            for _, audio in pending
        ]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.inference_mode():
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    generated = model.generate(
                        batch["input_features"],
                        max_length=GENERATION_MAX_LENGTH,
                    )
            else:
                generated = model.generate(
                    batch["input_features"],
                    max_length=GENERATION_MAX_LENGTH,
                )

        texts = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for (record, _), text in zip(pending, texts):
            outputs.append((record.key, language_code, text.strip()))

    if skipped_audio:
        print(f"  skipped {skipped_audio} clip(s) with unreadable audio", flush=True)
    return outputs


def write_submission(
    rows: list[tuple[str, str, str]],
    output_path: Path,
    *,
    id_column: str,
    language_column: str,
    text_column: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.DataFrame(rows, columns=[id_column, language_column, text_column])
        frame.to_parquet(output_path, index=False)
        return
    if suffix != ".csv":
        raise ValueError(f"Unsupported output format: {output_path.suffix} (use .csv or .parquet)")

    import csv

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([id_column, language_column, text_column])
        writer.writerows(rows)


def collect_test_records(args: argparse.Namespace) -> list[TrainingRecord]:
    if args.test_source == "kaggle_nt":
        test_root = args.kaggle_test_root.resolve()
        if not test_root.is_dir():
            print(f"ERROR: Kaggle test root not found: {test_root}", file=sys.stderr)
            return []
        return collect_kaggle_nt_test_records(test_root, max_samples=args.max_samples)

    return collect_records(
        work_dir=args.work_dir.resolve(),
        split="test",
        swahili_split=args.swahili_split,
        anv_split=args.anv_split,
        swahili_domains=tuple(DOMAINS),
        anv_languages=tuple(COMPETITION_ANV_LANGUAGES),
        include_swahili=True,
        include_anv=True,
        skip_maasai_scripted_train=False,
        require_transcript=False,
        max_samples=args.max_samples,
    )


def order_submission_rows(
    rows: list[tuple[str, str, str]],
    *,
    template: list[tuple[str, str]] | None,
) -> list[tuple[str, str, str]]:
    if not template:
        rows.sort(key=lambda item: item[0])
        return rows

    by_id = {clip_id: (language, transcription) for clip_id, language, transcription in rows}
    ordered: list[tuple[str, str, str]] = []
    missing_ids: list[str] = []
    for clip_id, language in template:
        prediction = by_id.pop(clip_id, None)
        if prediction is None:
            missing_ids.append(clip_id)
            ordered.append((clip_id, language, FAILED_AUDIO_PLACEHOLDER))
            continue
        predicted_language, transcription = prediction
        ordered.append((clip_id, language, transcription))

    if missing_ids:
        print(
            f"WARNING: {len(missing_ids)} ID(s) from sample_submission missing from predictions.",
            file=sys.stderr,
        )
    if by_id:
        print(
            f"WARNING: {len(by_id)} predicted ID(s) not in sample_submission; appending at end.",
            file=sys.stderr,
        )
        ordered.extend(
            (clip_id, language, transcription)
            for clip_id, (language, transcription) in sorted(by_id.items(), key=lambda item: item[0])
        )
    return ordered


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()

    test_records = collect_test_records(args)
    if not test_records:
        if args.test_source == "kaggle_nt":
            print(
                "ERROR: no test records found. Download anv-test-data-nt to --kaggle-test-root.",
                file=sys.stderr,
            )
        else:
            print("ERROR: no test records found. Check split paths under WORK_DIR.", file=sys.stderr)
        return 1

    print(f"Test source: {args.test_source}")
    if args.test_source == "kaggle_nt":
        print(f"Kaggle test root: {args.kaggle_test_root.resolve()}")
    else:
        print(f"Work dir: {work_dir}")
        print(f"Swahili split: {args.swahili_split}")
        print(f"Anv split: {args.anv_split}")
    print(f"Test clips: {len(test_records)} — {summarize_records(test_records)}")
    if args.expected_rows and len(test_records) != args.expected_rows:
        print(
            f"WARNING: expected {args.expected_rows} rows, found {len(test_records)}. "
            "Submission may be rejected by the competition site.",
            file=sys.stderr,
        )

    if args.dry_run:
        return 0

    if not model_dir.is_dir():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = resolve_autocast_dtype(device, args.dtype)
    print(f"Device: {device}")
    print(f"Model: {model_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Audio workers: {args.audio_workers}")
    print(
        f"Inference dtype: {args.dtype}"
        + (f" (autocast {autocast_dtype})" if autocast_dtype is not None else " (fp32)")
    )
    print(f"Force language prompts: {args.force_language_prompts}")

    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.to(device)
    model.eval()

    submission_rows: list[tuple[str, str, str]] = []
    languages = sorted({record.language for record in test_records})
    for language in languages:
        subset = [record for record in test_records if record.language == language]
        print(f"Transcribing {language}: {len(subset)} clip(s)", flush=True)
        if args.force_language_prompts:
            set_forced_language_prompt(model, processor, language)
        else:
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
        lang_rows = transcribe_language_batch(
            model,
            processor,
            subset,
            batch_size=args.batch_size,
            max_audio_seconds=args.max_audio_seconds,
            audio_workers=args.audio_workers,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        submission_rows.extend(lang_rows)

    template = None
    if args.test_source == "kaggle_nt":
        template = load_submission_template(args.kaggle_test_root.resolve(), id_column=args.id_column)
        if template:
            print(f"Ordering rows to match sample_submission ({len(template)} IDs)")

    submission_rows = order_submission_rows(submission_rows, template=template)
    write_submission(
        submission_rows,
        output_path,
        id_column=args.id_column,
        language_column=args.language_column,
        text_column=args.text_column,
    )

    print(f"Wrote {len(submission_rows)} rows to {output_path}")
    if args.expected_rows and len(submission_rows) != args.expected_rows:
        print(
            f"WARNING: wrote {len(submission_rows)} rows; expected {args.expected_rows}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
