#!/usr/bin/env python3
"""Repair a competition submission CSV for Kaggle upload.

Fixes common rejection causes:
1. Blank lines from doubled CR/LF (``\\r\\r\\n``).
2. Empty transcription cells for clips with unreadable audio.
3. Missing ``language`` column or wrong header casing.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.whisper_dataset import (  # noqa: E402
    build_submission_language_map,
    load_submission_template,
    load_submission_template_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix submission CSV for Kaggle upload.")
    parser.add_argument("input", type=Path, help="Broken submission CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fixed CSV path (default: <input stem>_fixed.csv).",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=None,
        help="Path to sample_submission.csv from Kaggle (preferred for id/language order).",
    )
    parser.add_argument(
        "--kaggle-test-root",
        type=Path,
        default=None,
        help="Path to anv-test-data-nt if sample_submission is inside it.",
    )
    parser.add_argument(
        "--empty-placeholder",
        default=".",
        help="Replacement text for empty transcriptions (default: '.').",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if header is None:
        raise ValueError(f"empty file: {path}")

    lower_header = [name.strip().lower() for name in header]
    id_idx = lower_header.index("id") if "id" in lower_header else 0
    text_idx = (
        lower_header.index("transcription")
        if "transcription" in lower_header
        else (
            lower_header.index("prediction")
            if "prediction" in lower_header
            else (lower_header.index("text") if "text" in lower_header else 1)
        )
    )

    predictions: dict[str, str] = {}
    for line_no, fields in enumerate(reader, start=2):
        if not fields or all(not field.strip() for field in fields):
            continue
        if len(fields) <= max(id_idx, text_idx):
            raise ValueError(f"{path}:{line_no}: missing expected columns")
        clip_id = fields[id_idx].strip()
        transcription = fields[text_idx]
        if clip_id:
            predictions[clip_id] = transcription
    return predictions


def load_template(args: argparse.Namespace) -> list[tuple[str, str]] | None:
    if args.sample_submission is not None:
        template = load_submission_template_file(args.sample_submission.resolve())
        if template is None:
            raise ValueError(f"could not read id/language from {args.sample_submission}")
        return template
    if args.kaggle_test_root is not None:
        return load_submission_template(args.kaggle_test_root.resolve(), id_column="id")
    return None


def build_template_from_language_map(
    predictions: dict[str, str],
    language_map: dict[str, str],
) -> list[tuple[str, str]]:
    return [(clip_id, language_map[clip_id]) for clip_id in predictions if clip_id in language_map]


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_fixed.csv")
    )

    if not input_path.is_file():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 1

    predictions = read_predictions(input_path)
    placeholder = args.empty_placeholder
    filled = 0

    template = load_template(args)
    if template is None and args.kaggle_test_root is not None:
        language_map = build_submission_language_map(args.kaggle_test_root.resolve())
        template = build_template_from_language_map(predictions, language_map)

    if template is None:
        print(
            "ERROR: need --sample-submission or --kaggle-test-root to add language column.",
            file=sys.stderr,
        )
        return 1

    fixed_rows: list[tuple[str, str, str]] = []
    missing_predictions = 0
    for clip_id, language in template:
        transcription = predictions.get(clip_id, "")
        if not transcription.strip():
            transcription = placeholder
            if clip_id not in predictions:
                missing_predictions += 1
            else:
                filled += 1
        fixed_rows.append((clip_id, language, transcription.strip() if transcription != placeholder else placeholder))

    extra_ids = sorted(set(predictions) - {clip_id for clip_id, _ in template})
    if extra_ids:
        print(
            f"WARNING: {len(extra_ids)} prediction ID(s) not in template; appending using language map.",
            file=sys.stderr,
        )
        language_map = {}
        if args.kaggle_test_root is not None:
            language_map = build_submission_language_map(args.kaggle_test_root.resolve())
        for clip_id in extra_ids:
            language = language_map.get(clip_id)
            if not language:
                print(f"WARNING: no language for extra id {clip_id}; skipping.", file=sys.stderr)
                continue
            text = predictions[clip_id].strip() or placeholder
            fixed_rows.append((clip_id, language, text))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "language", "transcription"])
        writer.writerows(fixed_rows)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows:   {len(fixed_rows)}")
    print(f"Filled empty transcriptions: {filled}")
    if missing_predictions:
        print(f"Missing predictions for template IDs: {missing_predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
