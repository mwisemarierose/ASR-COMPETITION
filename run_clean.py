#!/usr/bin/env python3
"""
CLI entry point for the Afrivoice_Swahili cleaning pipeline.

Examples:
    # Point at your Orchard / local dataset copy
    python run_clean.py --dataset-root /path/to/Afrivoice_Swahili

    # Test one small split first
    python run_clean.py --dataset-root /path/to/Afrivoice_Swahili \\
        --domain agriculture --split dev

    # Verify manifests only (fast, no output manifests)
    python run_clean.py --dataset-root /path/to/Afrivoice_Swahili --verify-only

    # Skip librosa audio reads and trust manifest duration
    python run_clean.py --dataset-root /path/to/Afrivoice_Swahili --skip-audio-check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import DEFAULT_DATASET_ROOT, PipelineConfig
from src.ffmpeg_setup import configure_ffmpeg
from src.pipeline import AfrivoiceCleaningPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean DigitalUmuganda/Afrivoice_Swahili ASR dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to Afrivoice_Swahili root (or set DATASET_ROOT env var)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Where cleaned manifests are written (default: data/cleaned)",
    )
    parser.add_argument("--domain", choices=["agriculture", "education", "financial", "government", "health"])
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--verify-only", action="store_true", help="Only verify integrity")
    parser.add_argument("--dry-run", action="store_true", help="Verify only, alias for quick checks")
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="Do not open audio files; use manifest duration instead",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Drop rows with missing image prompt files",
    )
    parser.add_argument(
        "--extract-cache-root",
        type=Path,
        help="Where tar.xz archives are extracted (default: data/extracted)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Use existing extracted cache only; do not unpack archives",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract tar.xz archives even if cache exists",
    )
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional upper duration limit in seconds (default: no limit)",
    )
    parser.add_argument("--max-records", type=int, help="Limit rows per split (for testing)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.skip_audio_check:
        configure_ffmpeg()

    config = PipelineConfig(
        dataset_root=args.dataset_root,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        verify_audio=not args.skip_audio_check,
        verify_images=args.require_images,
        dry_run=args.dry_run,
        skip_extract=args.skip_extract,
        force_extract=args.force_extract,
        max_records=args.max_records,
    )
    if args.output_root:
        config.output_root = args.output_root
    if args.extract_cache_root:
        config.extract_cache_root = args.extract_cache_root

    pipeline = AfrivoiceCleaningPipeline(config)
    return pipeline.run(
        domain=args.domain,
        split=args.split,
        verify_only=args.verify_only or args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
