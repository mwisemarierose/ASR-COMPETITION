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
    --kaggle-test-root /project/.../datasets/anv-test-data-nt \\
    --sample-submission /path/to/sample_submission.csv

Output format matches sample_submission.csv: id, language, transcription
(e.g. bpZJK6vvnq_18Nov2024071546GMT_1731914146936.wav, kik, ...).
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import DOMAINS, SAMPLE_RATE  # noqa: E402
from src.whisper_dataset import (  # noqa: E402
    COMPETITION_ANV_LANGUAGES,
    TrainingRecord,
    apply_kaggle_nt_submission_ids,
    collect_kaggle_nt_test_records,
    collect_records,
    load_submission_template,
    load_submission_template_file,
    set_forced_language_prompt,
    submission_language_code,
    summarize_records,
    try_load_record_audio,
)

MAX_AUDIO_SECONDS = 30.0
GENERATION_MAX_LENGTH = 225
# Kaggle rejects empty transcription cells as null values.
FAILED_AUDIO_PLACEHOLDER = "."


def submission_clip_id(record: TrainingRecord) -> str:
    """Return the Kaggle ``id`` column value written to the submission CSV."""
    return record.key


def resolve_submission_template(args: argparse.Namespace) -> list[tuple[str, str]] | None:
    """
    Load sample_submission.csv (id, language) rows — the competition upload format.

    For kaggle_nt this is required. IDs look like
    ``bpZJK6vvnq_18Nov2024071546GMT_1731914146936.wav``; language codes are
    ``kik``, ``swa``, etc.
    """
    if args.sample_submission is not None:
        template = load_submission_template_file(args.sample_submission.resolve())
        if template is None:
            print(
                f"ERROR: could not read id/language from {args.sample_submission}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return template

    if args.test_source != "kaggle_nt":
        return None

    template = load_submission_template(
        args.kaggle_test_root.resolve(),
        id_column=args.id_column,
        language_column=args.language_column,
    )
    if template is None:
        print(
            "ERROR: sample_submission.csv not found.\n"
            "  Download it from the Kaggle competition page, or place it under\n"
            f"  {args.kaggle_test_root.resolve()}/sample_submission.csv\n"
            "  Or pass: --sample-submission /path/to/sample_submission.csv",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return template


def validate_template_coverage(
    template: list[tuple[str, str]],
    records: list[TrainingRecord],
    *,
    id_column: str,
) -> None:
    """Ensure every sample_submission id has matching test audio."""
    record_ids = {submission_clip_id(record) for record in records}
    template_ids = [clip_id for clip_id, _language in template]
    missing = [clip_id for clip_id in template_ids if clip_id not in record_ids]
    if missing:
        sample = ", ".join(missing[:3])
        print(
            f"ERROR: {len(missing)} sample_submission id(s) missing test audio; "
            f"examples: {sample}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    extra = sorted(record_ids - set(template_ids))
    if extra:
        sample = ", ".join(extra[:3])
        print(
            f"WARNING: {len(extra)} test clip(s) not in sample_submission; "
            f"will be ignored. Examples: {sample}",
            file=sys.stderr,
        )

    print(
        f"Submission template: {len(template):,} rows "
        f"({id_column} + language from sample_submission.csv)",
        flush=True,
    )


def build_submission_rows_from_template(
    template: list[tuple[str, str]],
    predictions_by_id: dict[str, str],
) -> list[tuple[str, str, str]]:
    """
    Build competition CSV rows: id, language, transcription.

    Matches remap_submission_ids.py output — ids and language come from
    sample_submission; transcriptions from inference keyed by id.
    """
    rows: list[tuple[str, str, str]] = []
    missing = 0
    for clip_id, language in template:
        transcription = predictions_by_id.get(clip_id)
        if transcription is None or not str(transcription).strip():
            missing += 1
            rows.append((clip_id, language, FAILED_AUDIO_PLACEHOLDER))
            continue
        text = str(transcription).strip() or FAILED_AUDIO_PLACEHOLDER
        rows.append((clip_id, language, text))

    if missing:
        print(
            f"WARNING: {missing} sample_submission id(s) missing predictions; "
            f"filled with '{FAILED_AUDIO_PLACEHOLDER}'.",
            file=sys.stderr,
        )
    return rows


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
        "--sample-submission",
        type=Path,
        default=(
            Path(os.environ["SAMPLE_SUBMISSION"])
            if os.environ.get("SAMPLE_SUBMISSION")
            else None
        ),
        help="Path to sample_submission.csv from Kaggle (defines id, language, row order).",
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
    audio_executor: ThreadPoolExecutor | None = None,
) -> list[tuple[TrainingRecord, dict[str, Any] | None]]:
    if audio_workers <= 1 or len(batch_records) <= 1:
        return [
            (record, load_audio_for_record(record, max_audio_seconds))
            for record in batch_records
        ]

    loader = lambda record: load_audio_for_record(record, max_audio_seconds)
    if audio_executor is not None:
        audios = audio_executor.map(loader, batch_records)
    else:
        with ThreadPoolExecutor(max_workers=min(audio_workers, len(batch_records))) as pool:
            audios = pool.map(loader, batch_records)
    return list(zip(batch_records, audios))


def iter_transcription_batches(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    records: list[TrainingRecord],
    *,
    batch_size: int,
    max_audio_seconds: float,
    audio_workers: int,
    audio_executor: ThreadPoolExecutor | None,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    id_to_language: dict[str, str] | None = None,
) -> Iterator[list[tuple[str, str, str]]]:
    """Yield submission rows batch-by-batch so callers can flush to disk immediately."""

    def row_language(record: TrainingRecord) -> str:
        clip_id = submission_clip_id(record)
        if id_to_language is not None:
            return id_to_language[clip_id]
        return submission_language_code(record.language)

    skipped_audio = 0

    for start in tqdm(range(0, len(records), batch_size), desc="  batches", leave=False):
        batch_records = records[start : start + batch_size]
        batch_rows: list[tuple[str, str, str]] = []
        pending: list[tuple[TrainingRecord, dict[str, Any]]] = []

        for record, audio in load_batch_audio(
            batch_records,
            max_audio_seconds,
            audio_workers,
            audio_executor=audio_executor,
        ):
            clip_id = submission_clip_id(record)
            language_code = row_language(record)
            if audio is None:
                skipped_audio += 1
                batch_rows.append((clip_id, language_code, FAILED_AUDIO_PLACEHOLDER))
                continue
            pending.append((record, audio))

        if pending:
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
            del input_features

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
            del batch, generated
            for (record, _), text in zip(pending, texts):
                batch_rows.append((submission_clip_id(record), row_language(record), text.strip()))
            del pending

        if batch_rows:
            yield batch_rows
            del batch_rows

    if skipped_audio:
        print(f"  skipped {skipped_audio} clip(s) with unreadable audio", flush=True)


def append_submission_rows(
    rows: list[tuple[str, str, str]],
    output_path: Path,
    *,
    id_column: str,
    language_column: str,
    text_column: str,
    write_header: bool,
) -> None:
    """Append one inference batch to a partial CSV (streaming submission generation)."""
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if write_header:
            writer.writerow([id_column, language_column, text_column])
        writer.writerows(rows)


def load_submission_transcriptions(
    path: Path,
    *,
    id_column: str,
    text_column: str,
) -> dict[str, str]:
    """Load partial/final CSV into id -> transcription."""
    import csv

    by_id: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clip_id = str(row[id_column]).strip()
            if clip_id:
                by_id[clip_id] = str(row[text_column])
    return by_id


def load_submission_predictions(
    path: Path,
    *,
    id_column: str,
    language_column: str,
    text_column: str,
) -> dict[str, tuple[str, str]]:
    """Load partial CSV into id -> (language, transcription). Used when no template."""
    import csv

    by_id: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clip_id = str(row[id_column]).strip()
            if clip_id:
                by_id[clip_id] = (
                    str(row[language_column]).strip(),
                    str(row[text_column]),
                )
    return by_id


def write_submission(
    rows: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    """Write competition CSV: id, language, transcription (matches sample_submission format)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.DataFrame(rows, columns=["id", "language", "transcription"])
        frame.to_parquet(output_path, index=False)
        return
    if suffix != ".csv":
        raise ValueError(f"Unsupported output format: {output_path.suffix} (use .csv or .parquet)")

    import csv

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "language", "transcription"])
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
        require_transcript=False,
        max_samples=args.max_samples,
    )


def order_submission_predictions(
    by_id: dict[str, tuple[str, str]],
    *,
    template: list[tuple[str, str]] | None,
) -> list[tuple[str, str, str]]:
    if not template:
        return [
            (clip_id, language, transcription)
            for clip_id, (language, transcription) in sorted(by_id.items(), key=lambda item: item[0])
        ]

    ordered: list[tuple[str, str, str]] = []
    missing_ids: list[str] = []
    for clip_id, language in template:
        prediction = by_id.pop(clip_id, None)
        if prediction is None:
            missing_ids.append(clip_id)
            ordered.append((clip_id, language, FAILED_AUDIO_PLACEHOLDER))
            continue
        _predicted_language, transcription = prediction
        ordered.append((clip_id, language, transcription))

    if missing_ids:
        print(
            f"WARNING: {len(missing_ids)} ID(s) from sample_submission missing from predictions.",
            file=sys.stderr,
        )
    if by_id:
        print(
            f"WARNING: {len(by_id)} predicted ID(s) not in sample_submission; dropping.",
            file=sys.stderr,
        )
    return ordered


def order_submission_rows(
    rows: list[tuple[str, str, str]],
    *,
    template: list[tuple[str, str]] | None,
) -> list[tuple[str, str, str]]:
    by_id = {clip_id: (language, transcription) for clip_id, language, transcription in rows}
    return order_submission_predictions(by_id, template=template)


def _partial_submission_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial.csv")


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()

    template = resolve_submission_template(args)

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

    id_to_language: dict[str, str] | None = None
    if args.test_source == "kaggle_nt":
        test_root = args.kaggle_test_root.resolve()
        test_records = apply_kaggle_nt_submission_ids(test_records, test_root)
        if template is not None:
            validate_template_coverage(
                template,
                test_records,
                id_column=args.id_column,
            )
            id_to_language = {clip_id: language for clip_id, language in template}
            template_ids = {clip_id for clip_id, _language in template}
            test_records = [
                record for record in test_records if submission_clip_id(record) in template_ids
            ]

    print(f"Test source: {args.test_source}")
    if args.test_source == "kaggle_nt":
        print(f"Kaggle test root: {args.kaggle_test_root.resolve()}")
        if args.sample_submission is not None:
            print(f"Sample submission: {args.sample_submission.resolve()}")
    else:
        print(f"Work dir: {work_dir}")
        print(f"Swahili split: {args.swahili_split}")
        print(f"Anv split: {args.anv_split}")
    print(f"Test clips: {len(test_records)} — {summarize_records(test_records)}")
    if template is not None and args.expected_rows and len(template) != args.expected_rows:
        print(
            f"WARNING: sample_submission has {len(template)} rows; expected {args.expected_rows}.",
            file=sys.stderr,
        )
    elif args.expected_rows and len(test_records) != args.expected_rows and template is None:
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

    partial_path = _partial_submission_path(output_path)
    if partial_path.exists():
        partial_path.unlink()

    row_count = 0
    write_header = True
    languages = sorted({record.language for record in test_records})
    transcribe_kwargs = dict(
        batch_size=args.batch_size,
        max_audio_seconds=args.max_audio_seconds,
        audio_workers=args.audio_workers,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    column_kwargs = dict(
        id_column=args.id_column,
        language_column=args.language_column,
        text_column=args.text_column,
    )
    executor_ctx = (
        ThreadPoolExecutor(max_workers=args.audio_workers)
        if args.audio_workers > 1
        else nullcontext()
    )
    print(f"Streaming partial rows to {partial_path}", flush=True)
    with executor_ctx as audio_executor:
        for language in languages:
            subset = [record for record in test_records if record.language == language]
            print(f"Transcribing {language}: {len(subset)} clip(s)", flush=True)
            if args.force_language_prompts:
                set_forced_language_prompt(model, processor, language)
            else:
                model.config.forced_decoder_ids = None
                model.config.suppress_tokens = []
            for batch_rows in iter_transcription_batches(
                model,
                processor,
                subset,
                audio_executor=audio_executor if args.audio_workers > 1 else None,
                id_to_language=id_to_language,
                **transcribe_kwargs,
            ):
                append_submission_rows(
                    batch_rows,
                    partial_path,
                    write_header=write_header,
                    **column_kwargs,
                )
                write_header = False
                row_count += len(batch_rows)

    print(f"Finalizing submission from {row_count} streamed row(s)", flush=True)
    predictions_by_id = load_submission_transcriptions(
        partial_path,
        id_column=args.id_column,
        text_column=args.text_column,
    )
    if template is not None:
        submission_rows = build_submission_rows_from_template(template, predictions_by_id)
    else:
        by_id = load_submission_predictions(partial_path, **column_kwargs)
        submission_rows = order_submission_predictions(by_id, template=None)
    del predictions_by_id
    write_submission(submission_rows, output_path)
    partial_path.unlink(missing_ok=True)

    print(f"Wrote {len(submission_rows)} rows to {output_path}")
    if submission_rows:
        sample_id, sample_lang, _ = submission_rows[0]
        print(f"Sample row: {sample_id} ({sample_lang})")
    if args.expected_rows and len(submission_rows) != args.expected_rows:
        print(
            f"WARNING: wrote {len(submission_rows)} rows; expected {args.expected_rows}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
