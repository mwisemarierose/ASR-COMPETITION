#!/usr/bin/env python3
"""
Remap competition submission IDs from legacy composite keys to Parquet ``id`` values.

Older generate_submission runs used make_parquet_record_key() (e.g.
``recorder_uuid_test_unscripted_024_000347``) instead of the Parquet ``id`` column
(e.g. ``bpZJK6vvnq_18Nov2024071546GMT_1731914146936.wav``).

This script fixes an existing submission CSV without re-running GPU inference.

Usage (Orchard):
  python scripts/remap_submission_ids.py \\
    --input /project/.../submission_checkpoint-2500_job126124.csv \\
    --output /project/.../submission_fixed_ids.csv \\
    --kaggle-test-root /project/.../datasets/anv-test-data-nt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.audio_utils import (  # noqa: E402
    PARQUET_AUDIO_COLUMNS,
    make_parquet_record_key,
    resolve_parquet_id_column,
)
from src.whisper_dataset import (  # noqa: E402
    KAGGLE_NT_LANGUAGE_DIRS,
    KAGGLE_NT_STYLE_DIRS,
    submission_language_code,
)

FAILED_AUDIO_PLACEHOLDER = "."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap legacy submission IDs to Parquet id column values.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Submission CSV with legacy composite ids.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fixed submission CSV path.",
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
        help="Root of digitalumuganda/anv-test-data-nt.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=41733,
        help="Expected row count (0 = skip check).",
    )
    return parser.parse_args()


def _parquet_has_audio_column(schema_names: list[str]) -> bool:
    return any(name in schema_names for name in PARQUET_AUDIO_COLUMNS)


def _resolve_kaggle_style_dir(lang_dir: Path, style_name: str) -> Path | None:
    for candidate in (style_name, style_name.lower(), style_name.capitalize()):
        path = lang_dir / candidate
        if path.is_dir():
            return path
    return None


def _cell_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_legacy_id_remap(test_root: Path) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """
    Return:
      - legacy_key -> correct Parquet id
      - ordered rows (correct_id, submission_language_code, legacy_key) in test iteration order
    """
    test_root = test_root.resolve()
    if not test_root.is_dir():
        raise FileNotFoundError(f"Kaggle test root not found: {test_root}")

    legacy_to_correct: dict[str, str] = {}
    ordered_rows: list[tuple[str, str, str]] = []
    duplicate_legacy: list[str] = []

    for dir_name, language in sorted(KAGGLE_NT_LANGUAGE_DIRS.items()):
        lang_dir = test_root / dir_name
        if not lang_dir.is_dir():
            continue
        lang_code = submission_language_code(language)

        for style_name in KAGGLE_NT_STYLE_DIRS:
            style_dir = _resolve_kaggle_style_dir(lang_dir, style_name)
            if style_dir is None:
                continue

            for parquet_path in sorted(style_dir.glob("*.parquet")):
                schema_names = pq.read_schema(parquet_path).names
                if not _parquet_has_audio_column(schema_names):
                    continue
                if "id" not in schema_names:
                    raise RuntimeError(f"Parquet file missing required 'id' column: {parquet_path}")

                id_col = resolve_parquet_id_column(schema_names)
                read_columns = ["id"]
                if id_col and id_col not in read_columns:
                    read_columns.append(id_col)
                for extra_id_col in ("mediaPathId", "media_path_id"):
                    if extra_id_col in schema_names and extra_id_col not in read_columns:
                        read_columns.append(extra_id_col)

                table = pq.read_table(parquet_path, columns=read_columns)
                for row_index in range(table.num_rows):
                    correct_id = _cell_str(table.column("id")[row_index].as_py())
                    if not correct_id:
                        raise RuntimeError(
                            f"Empty id at {parquet_path}:{row_index}",
                        )

                    recorder_id = None
                    media_path_id = None
                    if id_col:
                        recorder_id = _cell_str(table.column(id_col)[row_index].as_py())
                    if "mediaPathId" in read_columns:
                        media_path_id = _cell_str(table.column("mediaPathId")[row_index].as_py())
                    elif "media_path_id" in read_columns:
                        media_path_id = _cell_str(table.column("media_path_id")[row_index].as_py())

                    legacy_key = make_parquet_record_key(
                        recorder_id,
                        parquet_path,
                        row_index,
                        media_path_id,
                    )

                    previous = legacy_to_correct.get(legacy_key)
                    if previous is not None and previous != correct_id:
                        duplicate_legacy.append(legacy_key)
                    legacy_to_correct[legacy_key] = correct_id
                    ordered_rows.append((correct_id, lang_code, legacy_key))

    if duplicate_legacy:
        sample = ", ".join(duplicate_legacy[:5])
        raise RuntimeError(
            f"Legacy key collision(s) ({len(duplicate_legacy)}); examples: {sample}",
        )

    return legacy_to_correct, ordered_rows


def read_submission_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty submission file: {path}")
        for row_no, row in enumerate(reader, start=2):
            clip_id = str(row.get("id", "")).strip()
            if not clip_id:
                continue
            transcription = row.get("transcription", "")
            if clip_id in predictions:
                print(
                    f"WARNING: duplicate legacy id in input ({clip_id}) at row {row_no}; keeping first.",
                    file=sys.stderr,
                )
                continue
            predictions[clip_id] = transcription
    return predictions


def write_submission(
    rows: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "language", "transcription"])
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    test_root = args.kaggle_test_root.resolve()

    if not input_path.is_file():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Test:   {test_root}")

    legacy_to_correct, ordered_rows = build_legacy_id_remap(test_root)
    predictions = read_submission_predictions(input_path)

    print(f"Legacy id map: {len(legacy_to_correct):,} clips")
    print(f"Input predictions: {len(predictions):,} rows")

    missing_legacy = 0
    unused_predictions = set(predictions)
    fixed_rows: list[tuple[str, str, str]] = []

    for correct_id, lang_code, legacy_key in ordered_rows:
        transcription = predictions.get(legacy_key)
        if transcription is None:
            missing_legacy += 1
            fixed_rows.append((correct_id, lang_code, FAILED_AUDIO_PLACEHOLDER))
            continue
        unused_predictions.discard(legacy_key)
        text = transcription.strip() or FAILED_AUDIO_PLACEHOLDER
        fixed_rows.append((correct_id, lang_code, text))

    if missing_legacy:
        print(
            f"WARNING: {missing_legacy} clip(s) missing from input submission; filled with '{FAILED_AUDIO_PLACEHOLDER}'.",
            file=sys.stderr,
        )
    if unused_predictions:
        sample = ", ".join(sorted(unused_predictions)[:5])
        print(
            f"WARNING: {len(unused_predictions)} input id(s) did not match any legacy key; ignored. "
            f"Examples: {sample}",
            file=sys.stderr,
        )

    write_submission(fixed_rows, output_path)

    print(f"Wrote {len(fixed_rows):,} rows to {output_path}")
    if args.expected_rows and len(fixed_rows) != args.expected_rows:
        print(
            f"WARNING: expected {args.expected_rows} rows, wrote {len(fixed_rows)}.",
            file=sys.stderr,
        )

    print("Sample fixed ids:")
    for correct_id, lang_code, _ in fixed_rows[:3]:
        print(f"  {correct_id} ({lang_code})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
