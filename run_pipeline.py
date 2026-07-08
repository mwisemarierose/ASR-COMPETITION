#!/usr/bin/env python3
"""
Run the full Afrivoice_Swahili ASR data pipeline.

Steps:
    1. clean      — remove bad rows, normalize transcripts
    2. preprocess — resample audio to 16 kHz WAV
    3. extract    — generate 80-bin log-mel spectrograms
    4. validate   — verify feature files across all splits

Usage:
    python run_pipeline.py --dataset-root /path/to/Afrivoice_Swahili
    python run_pipeline.py --dataset-root tests/fixtures --domain agriculture --split dev
    python run_pipeline.py --step clean --domain agriculture --split dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import DEFAULT_DATASET_ROOT, PipelineConfig
from src.ffmpeg_setup import configure_ffmpeg, ffmpeg_status
from src.feature_validator import FeatureValidator
from src.features import AfrivoiceFeaturePipeline
from src.pipeline import AfrivoiceCleaningPipeline
from src.preprocessing import AfrivoicePreprocessingPipeline

STEPS = ("clean", "preprocess", "extract", "validate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Afrivoice ASR pipeline")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Put all large outputs here (extracted, cleaned, processed, features). "
        "Use community/project storage on HPC to avoid home disk quota.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--extract-cache-root", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--domain", choices=["agriculture", "education", "financial", "government", "health"])
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument(
        "--step",
        choices=STEPS,
        action="append",
        help="Run only specific step(s); default runs all",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-audio-check", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-alignment-check", action="store_true")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the pre-clean full manifest scan (faster for large train splits)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for the clean step (use SLURM --cpus-per-task value)",
    )
    parser.add_argument("--max-records", type=int)
    return parser


def build_config(args: argparse.Namespace) -> PipelineConfig:
    if args.work_dir:
        config = PipelineConfig.with_work_dir(
            args.work_dir,
            dataset_root=args.dataset_root,
            verify_audio=not args.skip_audio_check,
            verify_alignment=not args.skip_alignment_check,
            skip_extract=args.skip_extract,
            force_extract=args.force_extract,
            skip_verify=args.skip_verify,
            workers=max(1, args.workers),
            max_records=args.max_records,
        )
    else:
        config = PipelineConfig(
            dataset_root=args.dataset_root,
            verify_audio=not args.skip_audio_check,
            verify_alignment=not args.skip_alignment_check,
            skip_extract=args.skip_extract,
            force_extract=args.force_extract,
            skip_verify=args.skip_verify,
            workers=max(1, args.workers),
            max_records=args.max_records,
        )

    if args.output_root:
        config.output_root = args.output_root
    if args.extract_cache_root:
        config.extract_cache_root = args.extract_cache_root
    if args.processed_root:
        config.processed_root = args.processed_root
    if args.features_dir:
        config.features_dir = args.features_dir
    return config


def main() -> int:
    args = build_parser().parse_args()
    config = build_config(args)
    steps = args.step or STEPS

    needs_ffmpeg = any(step in steps for step in ("clean", "preprocess")) and not args.skip_audio_check
    if needs_ffmpeg or "preprocess" in steps:
        ffmpeg_path = configure_ffmpeg()
        if not ffmpeg_path and "preprocess" in steps:
            print(ffmpeg_status())
            print("\nPreprocessing .webm files requires ffmpeg.")
            return 1
        if ffmpeg_path:
            print(f"Using ffmpeg: {ffmpeg_path}")

    if args.work_dir:
        print(f"Work directory: {args.work_dir.resolve()}")
        print(f"  extracted: {config.extract_cache_root}")
        print(f"  cleaned:   {config.output_root}")
        print(f"  processed: {config.processed_root}")
        print(f"  features:  {config.features_dir}")

    if "clean" in steps:
        print(f"\n{'=' * 60}\nCLEAN\n{'=' * 60}")
        code = AfrivoiceCleaningPipeline(config).run(
            domain=args.domain,
            split=args.split,
            verify_only=args.verify_only,
        )
        if code != 0:
            return code
        if args.verify_only:
            return 0

    if "preprocess" in steps:
        print(f"\n{'=' * 60}\nPREPROCESS\n{'=' * 60}")
        code = AfrivoicePreprocessingPipeline(config).run(domain=args.domain, split=args.split)
        if code != 0:
            return code

    feature_pipeline = AfrivoiceFeaturePipeline(config)

    if "extract" in steps:
        print(f"\n{'=' * 60}\nEXTRACT FEATURES\n{'=' * 60}")
        code = feature_pipeline.run_extract(domain=args.domain, split=args.split)
        if code != 0:
            return code

    if "validate" in steps:
        print(f"\n{'=' * 60}\nVALIDATE FEATURES\n{'=' * 60}")
        code = FeatureValidator(config).run(domain=args.domain, split=args.split)
        if code != 0:
            return code

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
