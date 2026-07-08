"""
Main OOP cleaning pipeline for DigitalUmuganda/Afrivoice_Swahili.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .archive_extractor import ExtractionReport, TarXzArchiveExtractor
from .clean_workers import process_clean_batch
from .config import PipelineConfig
from .discovery import DatasetDiscovery
from .filters import RecordFilter
from .manifest_loader import ManifestChunkLoader
from .media_resolver import MediaResolver
from .models import AfrivoiceRecord, FilterStats, SplitContext, VerifyReport
from .validators import AudioFileValidator, TranscriptValidator


class AfrivoiceCleaningPipeline:
    """
    Verify and clean one or many Afrivoice domain/split folders.

    Designed for large on-disk datasets:
    - streams manifest_*.jsonl line by line
    - processes one split folder at a time
    - writes cleaned JSONL without copying audio
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.discovery = DatasetDiscovery(config)
        self.extractor = TarXzArchiveExtractor(config)

    def run(
        self,
        domain: str | None = None,
        split: str | None = None,
        verify_only: bool = False,
    ) -> int:
        targets = list(self.discovery.iter_targets(domain=domain, split=split))
        if not targets:
            print(f"No split folders found under {self.config.dataset_root}")
            return 1

        reports: list[dict[str, Any]] = []
        for context in targets:
            print(f"\n{'=' * 60}")
            print(f"{context.domain} / {context.split}")
            print(f"Folder: {context.folder}")

            extraction_report = self._prepare_split(context)
            if self.config.skip_verify:
                verify_report = self._quick_verify(context, extraction_report)
            else:
                verify_report = self.verify_split(context, extraction_report)
            self._print_verify(verify_report)
            if not verify_report.ok:
                print("Aborting this split — fix raw data first.")
                return 1

            reports.append({"verify": verify_report.to_dict()})
            if verify_only or self.config.dry_run:
                continue

            clean_report = self.clean_split(context)
            reports[-1]["clean"] = clean_report
            self._print_clean(clean_report)

        if not self.config.dry_run:
            self._write_report(reports)

        if verify_only:
            print("\nVerification complete.")
        elif self.config.dry_run:
            print("\nDry run complete (no cleaned manifests written).")
        else:
            print("\nCleaning complete.")
        return 0

    def _prepare_split(self, context) -> ExtractionReport | None:
        if not context.audio_archives:
            return None

        print("  extracting audio archives...")
        report = self.extractor.prepare_split(context)
        if report:
            status = "cached" if report.skipped else "extracted"
            print(
                f"    audio: {status} "
                f"({len(report.archives)} archive(s), {report.extracted_files} files)"
            )
            if report.target_dir:
                print(f"      -> {report.target_dir}")
        return report

    def _quick_verify(
        self,
        context: SplitContext,
        extraction_report: ExtractionReport | None = None,
    ) -> VerifyReport:
        """Fast structural checks without scanning every manifest row."""
        report = VerifyReport(
            domain=context.domain,
            split=context.split,
            folder=context.folder,
        )
        if not context.folder.is_dir():
            report.issues.append(f"Directory not found: {context.folder}")
            return report
        if not context.manifest_paths:
            report.issues.append("No manifest_*.jsonl files found")
            return report

        report.manifests = [path.name for path in context.manifest_paths]
        report.audio_archives = [path.name for path in context.audio_archives]
        if extraction_report:
            report.extraction = extraction_report.to_dict()
        return report

    def verify_split(self, context, extraction_report: ExtractionReport | None = None) -> VerifyReport:
        report = VerifyReport(
            domain=context.domain,
            split=context.split,
            folder=context.folder,
        )

        if not context.folder.is_dir():
            report.issues.append(f"Directory not found: {context.folder}")
            return report

        if not context.manifest_paths:
            report.issues.append("No manifest_*.jsonl files found")
            return report

        report.manifests = [path.name for path in context.manifest_paths]
        report.audio_archives = [path.name for path in context.audio_archives]
        if extraction_report:
            report.extraction = extraction_report.to_dict()
        transcript_validator = TranscriptValidator()
        audio_validator = AudioFileValidator()
        loader = ManifestChunkLoader(context.manifest_paths)

        for record in loader:
            report.rows += 1
            ok, reason = transcript_validator.validate(record, context)
            if not ok:
                report.empty_transcripts += 1

            ok, reason = audio_validator.validate(record, context)
            if not ok:
                report.missing_audio += 1

            if self.config.max_records and report.rows >= self.config.max_records:
                break

        return report

    def clean_split(self, context: SplitContext) -> dict[str, Any]:
        self._prepare_audio_index(context)
        if self.config.workers > 1:
            return self._clean_split_parallel(context)
        return self._clean_split_sequential(context)

    def _prepare_audio_index(self, context: SplitContext) -> None:
        if context.audio_index is not None:
            return
        context.audio_index = MediaResolver.build_audio_index(context)
        print(f"  audio index: {len(context.audio_index)} file(s)")

    def _clean_split_sequential(self, context: SplitContext) -> dict[str, Any]:
        stats = FilterStats()
        output_dir = self._output_dir(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "manifest_cleaned.jsonl"

        record_filter = RecordFilter(self.config)
        loader = ManifestChunkLoader(context.manifest_paths)

        with output_path.open("w", encoding="utf-8") as handle:
            for record in tqdm(loader, desc=f"  {context.domain}/{context.split}"):
                stats.input_rows += 1
                cleaned, reason = record_filter.process(record, context)
                record_filter.bump_stat(stats, reason)

                if cleaned is not None:
                    handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

                if self.config.max_records and stats.input_rows >= self.config.max_records:
                    break

        return {
            "domain": context.domain,
            "split": context.split,
            "output_manifest": str(output_path),
            "stats": stats.to_dict(),
        }

    def _clean_split_parallel(self, context: SplitContext) -> dict[str, Any]:
        records = self._load_records(context)
        if not records:
            raise RuntimeError(f"No manifest rows found for {context.domain}/{context.split}")

        workers = min(self.config.workers, len(records))
        chunk_size = max(1, (len(records) + workers - 1) // workers)
        batches = [records[index : index + chunk_size] for index in range(0, len(records), chunk_size)]

        print(f"  cleaning with {workers} worker(s), {len(batches)} batch(es)")
        stats = FilterStats()
        output_dir = self._output_dir(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "manifest_cleaned.jsonl"

        batch_results: list = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_clean_batch, batch, self.config, context)
                for batch in batches
            ]
            for future in tqdm(futures, desc=f"  {context.domain}/{context.split}"):
                batch_results.extend(future.result())

        batch_results.sort(key=lambda item: item[0])
        seen_keys: set[str] = set()

        with output_path.open("w", encoding="utf-8") as handle:
            for _, cleaned, reason in batch_results:
                stats.input_rows += 1
                if cleaned is not None:
                    key = str(cleaned.get("key", ""))
                    if key and key in seen_keys:
                        cleaned = None
                        reason = "duplicate_key"
                    elif key:
                        seen_keys.add(key)

                self._bump_stat(stats, reason)
                if cleaned is not None:
                    handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

                if self.config.max_records and stats.input_rows >= self.config.max_records:
                    break

        return {
            "domain": context.domain,
            "split": context.split,
            "output_manifest": str(output_path),
            "stats": stats.to_dict(),
        }

    def _load_records(self, context: SplitContext) -> list[AfrivoiceRecord]:
        records = list(ManifestChunkLoader(context.manifest_paths))
        if self.config.max_records:
            records = records[: self.config.max_records]
        return records

    @staticmethod
    def _bump_stat(stats: FilterStats, reason: str | None) -> None:
        if reason is None:
            stats.kept += 1
            return
        if hasattr(stats, reason):
            setattr(stats, reason, getattr(stats, reason) + 1)

    def _output_dir(self, context: SplitContext) -> Path:
        return self.config.output_root / context.domain / context.split

    def _write_report(self, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.stats_dir / "cleaning_report.json"
        path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved report: {path}")

    @staticmethod
    def _print_verify(report: VerifyReport) -> None:
        print(f"  manifests: {', '.join(report.manifests) or 'none'}")
        if report.audio_archives:
            print(f"  audio archives: {', '.join(report.audio_archives)}")
        print(f"  rows: {report.rows}")
        print(f"  empty transcripts: {report.empty_transcripts}")
        print(f"  missing audio: {report.missing_audio}")
        for issue in report.issues:
            print(f"  ! {issue}")

    @staticmethod
    def _print_clean(report: dict[str, Any]) -> None:
        stats = report["stats"]
        print(
            f"  kept: {stats['kept']}/{stats['input_rows']} "
            f"(empty={stats['empty_transcript']}, missing={stats['missing_audio']}, "
            f"corrupt={stats['corrupt_audio']}, short={stats['too_short']}, "
            f"long={stats['too_long']}, dup={stats['duplicate_key']}, "
            f"misaligned={stats['misaligned']})"
        )
        print(f"  output: {report['output_manifest']}")
