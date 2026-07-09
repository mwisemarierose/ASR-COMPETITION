#!/usr/bin/env python3
"""
Run ASR data pipelines for Afrivoice Swahili or Anv-ke language datasets.

Afrivoice: clean -> preprocess -> extract
Anv-ke:    clean -> extract (Parquet streaming, no WAV cache)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import (
    AFRIVOICE_LANGUAGES,
    ANV_LANGUAGES,
    ANV_STYLES,
    DEFAULT_AFRIVOICE_MULTILANG_ROOT,
    DEFAULT_DATASET_ROOT,
    DOMAINS,
    SPLITS,
    PipelineConfig,
)
from src.features import FeaturePipeline
from src.ffmpeg_setup import configure_ffmpeg, ffmpeg_status
from src.pipeline import CleaningPipeline
from src.preprocessing import AfrivoicePreprocessingPipeline

AFRIVOICE_STEPS = ("clean", "preprocess", "extract")
ANV_STEPS = ("clean", "extract")
ALL_LANGUAGE_CHOICES = sorted(set(AFRIVOICE_LANGUAGES) | set(ANV_LANGUAGES))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ASR dataset pipelines")
    parser.add_argument(
        "--dataset-type",
        choices=("afrivoice", "anv"),
        default="afrivoice",
        help="afrivoice = DigitalUmuganda tar.xz + manifests; anv = Anv-ke Parquet languages",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--language",
        choices=ALL_LANGUAGE_CHOICES,
        help="Language slug. Afrivoice: swahili (default), somali, shona, ... Anv-ke: kalenjin, dholuo, ...",
    )
    parser.add_argument("--style", choices=ANV_STYLES, help="Anv-ke style: scripted or unscripted")
    parser.add_argument("--work-dir", type=Path, help="Community/project storage for large outputs")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--extract-cache-root", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--domain", choices=list(DOMAINS))
    parser.add_argument("--split", choices=list(SPLITS) + ["dev_test", "all"])
    parser.add_argument("--step", choices=AFRIVOICE_STEPS + ANV_STEPS, action="append")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-audio-check", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-alignment-check", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    return parser


def build_config(args: argparse.Namespace) -> PipelineConfig:
    dataset_root = args.dataset_root
    if dataset_root is None:
        if args.dataset_type == "anv":
            dataset_root = DEFAULT_AFRIVOICE_MULTILANG_ROOT
        elif args.language and args.language != "swahili":
            dataset_root = DEFAULT_AFRIVOICE_MULTILANG_ROOT
        else:
            dataset_root = DEFAULT_DATASET_ROOT

    common = {
        "dataset_type": args.dataset_type,
        "dataset_root": dataset_root,
        "language": args.language,
        "verify_audio": not args.skip_audio_check,
        "skip_verify": args.skip_verify,
        "workers": max(1, args.workers),
        "max_records": args.max_records,
    }

    if args.dataset_type == "anv":
        if not args.language:
            raise SystemExit("--language is required when --dataset-type anv")
    else:
        common["verify_alignment"] = not args.skip_alignment_check
        common["skip_extract"] = args.skip_extract
        common["force_extract"] = args.force_extract

    if args.work_dir:
        config = PipelineConfig.with_work_dir(args.work_dir, **common)
    else:
        config = PipelineConfig(**common)

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
    steps = args.step or (ANV_STEPS if config.is_anv else AFRIVOICE_STEPS)

    if config.is_anv:
        if args.domain:
            print("Note: --domain is ignored for Anv-ke datasets.")
        if args.work_dir:
            print(f"Work directory: {args.work_dir.resolve()}")
            print(f"  dataset:   {config.language_dataset_root()}")
            print(f"  cleaned:   {config.output_root}")
            print(f"  features:  {config.features_dir}")
    else:
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
        title = f"CLEAN ({config.language_name})" if config.is_anv else "CLEAN"
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        code = CleaningPipeline(config).run(
            domain=args.domain,
            split=args.split,
            style=args.style,
            verify_only=args.verify_only,
        )
        if code != 0:
            return code
        if args.verify_only:
            return 0

    if "preprocess" in steps:
        if config.is_anv:
            print("Note: preprocess is skipped for Anv-ke (audio stays in Parquet).")
        else:
            print(f"\n{'=' * 60}\nPREPROCESS\n{'=' * 60}")
            code = AfrivoicePreprocessingPipeline(config).run(domain=args.domain, split=args.split)
            if code != 0:
                return code

    if "extract" in steps:
        title = f"EXTRACT FEATURES ({config.language_name})" if config.is_anv else "EXTRACT FEATURES"
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        code = FeaturePipeline(config).run_extract(
            domain=args.domain,
            split=args.split,
            style=args.style,
        )
        if code != 0:
            return code

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
