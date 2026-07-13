"""
Cleaning pipeline for Afrivoice Swahili and Anv-ke Parquet datasets.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm

from .archive_extractor import ExtractionReport, TarXzArchiveExtractor
from .audio_utils import (
    audio_bytes_from_struct,
    duration_from_audio_bytes,
    make_parquet_record_key,
    resolve_parquet_id_column,
    resolve_parquet_text_column,
)
from .config import PipelineConfig
from .discovery import DatasetDiscovery
from .filters import RecordFilter
from .manifest_loader import (
    ManifestChunkLoader,
    enrich_csv_row,
    merge_meta_tables,
    transcript_from_csv_row,
)
from .media_resolver import MediaResolver
from .models import AfrivoiceRecord, FilterStats, SplitContext, VerifyReport
from .transcript_cleaner import clean_anv_transcript
from .validators import AudioFileValidator, TranscriptValidator

CleanSortKey = tuple[str, int]
CleanResult = tuple[CleanSortKey, dict[str, Any] | None, str | None]


@dataclass
class ParquetFilterStats:
    input_rows: int = 0
    empty_transcript: int = 0
    missing_audio: int = 0
    corrupt_audio: int = 0
    too_short: int = 0
    too_long: int = 0
    duplicate_key: int = 0
    kept: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "empty_transcript": self.empty_transcript,
            "missing_audio": self.missing_audio,
            "corrupt_audio": self.corrupt_audio,
            "too_short": self.too_short,
            "too_long": self.too_long,
            "duplicate_key": self.duplicate_key,
            "kept": self.kept,
        }


def _process_parquet_shard(
    parquet_path: str,
    style: str,
    language: str,
    split: str,
    meta_table: dict[str, dict[str, str]],
    min_duration_sec: float,
    max_duration_sec: float | None,
    verify_audio: bool,
    max_records: int | None,
) -> tuple[list[dict[str, Any]], ParquetFilterStats]:
    path = Path(parquet_path)
    schema_names = pq.read_schema(path).names
    audio_col = next(
        (name for name in ("audio", "bytes", "audio_bytes") if name in schema_names),
        None,
    )
    if audio_col is None:
        return [], ParquetFilterStats()

    text_col = resolve_parquet_text_column(schema_names)
    id_col = resolve_parquet_id_column(schema_names)
    columns = [audio_col]
    if text_col:
        columns.append(text_col)
    if id_col:
        columns.append(id_col)
    for extra_id_col in ("mediaPathId", "media_path_id"):
        if extra_id_col in schema_names and extra_id_col not in columns:
            columns.append(extra_id_col)

    table = pq.read_table(path, columns=columns)
    stats = ParquetFilterStats()
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for row_index in range(table.num_rows):
        stats.input_rows += 1
        if max_records and len(rows) >= max_records:
            break

        audio_value = table.column(audio_col)[row_index].as_py()
        audio_bytes = audio_bytes_from_struct(audio_value)
        if not audio_bytes:
            stats.missing_audio += 1
            continue

        recorder_id = None
        media_path_id = None
        if id_col:
            recorder_id = table.column(id_col)[row_index].as_py()
            if recorder_id is not None:
                recorder_id = str(recorder_id).strip()
        if "mediaPathId" in schema_names:
            value = table.column("mediaPathId")[row_index].as_py()
            if value is not None:
                media_path_id = str(value).strip()
        elif "media_path_id" in schema_names:
            value = table.column("media_path_id")[row_index].as_py()
            if value is not None:
                media_path_id = str(value).strip()

        meta = meta_table.get(recorder_id or media_path_id or "", {}) if (recorder_id or media_path_id) else {}
        transcript = ""
        if text_col:
            value = table.column(text_col)[row_index].as_py()
            transcript = str(value or "").strip()
        if not transcript and meta:
            transcript = transcript_from_csv_row(meta, style)
        if not transcript:
            stats.empty_transcript += 1
            continue

        duration_sec = meta.get("duration") if meta else None
        if duration_sec:
            try:
                duration_sec = float(duration_sec)
            except (TypeError, ValueError):
                duration_sec = None

        if duration_sec is None and verify_audio:
            try:
                duration_sec = duration_from_audio_bytes(audio_bytes)
            except Exception:
                stats.corrupt_audio += 1
                continue

        if duration_sec is not None and duration_sec < min_duration_sec:
            stats.too_short += 1
            continue
        if duration_sec is not None and max_duration_sec is not None and duration_sec > max_duration_sec:
            stats.too_long += 1
            continue

        key = make_parquet_record_key(recorder_id, path, row_index, media_path_id)
        if key in seen_keys:
            stats.duplicate_key += 1
            continue
        seen_keys.add(key)

        cleaned: dict[str, Any] = {
            "key": key,
            "transcript": clean_anv_transcript(transcript),
            "duration_sec": duration_sec,
            "language": language,
            "split": split,
            "style": style,
            "audio_source": {
                "type": "parquet",
                "path": str(path.resolve()),
                "row": row_index,
            },
        }
        if recorder_id:
            cleaned["recorder_uuid"] = recorder_id
        cleaned.update(enrich_csv_row(meta, style))
        rows.append(cleaned)
        stats.kept += 1

    return rows, stats


def _process_clean_batch(
    batch: list[AfrivoiceRecord],
    config: PipelineConfig,
    context: SplitContext,
) -> list[CleanResult]:
    """Filter and normalize a batch of manifest rows (no duplicate-key dedup)."""
    record_filter = RecordFilter(config)
    results: list[CleanResult] = []
    for record in batch:
        cleaned, reason = record_filter.process(record, context)
        sort_key = (str(record.source_manifest), record.line_number)
        results.append((sort_key, cleaned, reason))
    return results


class CleaningPipeline:
    """
    Verify and clean dataset splits for Afrivoice Swahili or Anv-ke Parquet data.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.discovery = DatasetDiscovery(config)
        self.extractor = TarXzArchiveExtractor(config)

    def run(
        self,
        domain: str | None = None,
        split: str | None = None,
        style: str | None = None,
        verify_only: bool = False,
    ) -> int:
        if self.config.is_anv:
            return self._run_anv(split=split, style=style, verify_only=verify_only)
        return self._run_afrivoice(domain=domain, split=split, verify_only=verify_only)

    def _run_afrivoice(
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

            clean_report = self.clean_split(context, extraction_report)
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
            report.issues.append("No manifest_*.json(l) files found")
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
            report.issues.append("No manifest_*.json(l) files found")
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

    def clean_split(self, context: SplitContext, extraction_report: ExtractionReport | None = None) -> dict[str, Any]:
        self._prepare_audio_index(context, extraction_report)
        if self.config.workers > 1:
            return self._clean_split_parallel(context)
        return self._clean_split_sequential(context)

    def _prepare_audio_index(
        self,
        context: SplitContext,
        extraction_report: ExtractionReport | None = None,
    ) -> None:
        if context.audio_index is not None:
            return
        context.audio_index = MediaResolver.build_audio_index(context)
        indexed_files = len({path.resolve() for path in context.audio_index.values()})
        print(f"  audio index: {indexed_files} file(s)")

        expected = extraction_report.extracted_files if extraction_report else 0
        if indexed_files == 0:
            context.audio_index = None
            search_dirs = MediaResolver._search_dirs(context)
            print("  ! warning: audio index is empty — falling back to per-file lookup")
            for directory in search_dirs:
                print(f"    search dir: {directory} (exists={directory.is_dir()})")
            if expected:
                print(
                    f"    cache marker reports {expected} extracted file(s) — "
                    "re-run without --skip-extract if the cache directory is empty"
                )
        elif expected and indexed_files < expected * 0.9:
            print(
                f"  ! warning: indexed {indexed_files} file(s) but cache marker "
                f"reports {expected} — cache may be incomplete"
            )

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
                executor.submit(_process_clean_batch, batch, self.config, context)
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
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for report in reports:
            domain = (
                report.get("clean", {}).get("domain")
                or report.get("verify", {}).get("domain")
            )
            if not domain:
                continue
            by_domain.setdefault(domain, []).append(report)

        saved_paths: list[Path] = []
        for domain, domain_reports in sorted(by_domain.items()):
            path = self.config.stats_dir / f"cleaning_report_{domain}.json"
            path.write_text(json.dumps(domain_reports, indent=2), encoding="utf-8")
            saved_paths.append(path)

        joined = ", ".join(str(path) for path in saved_paths)
        print(f"\nSaved report(s): {joined}")

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

    def _run_anv(
        self,
        split: str | None = None,
        style: str | None = None,
        verify_only: bool = False,
    ) -> int:
        targets = list(self.discovery.iter_targets(split=split, style=style))
        if not targets:
            root = self.config.language_dataset_root()
            print(f"No split/style folders found under {root}")
            return 1

        reports: list[dict[str, Any]] = []
        for context in targets:
            print(f"\n{'=' * 60}")
            print(f"{context.language} / {context.split} / {context.style}")
            print(f"Folder: {context.folder}")

            verify_report = self._verify_parquet_split(context)
            self._print_parquet_verify(verify_report)
            if not verify_report["ok"] and not self.config.skip_verify:
                print("Aborting this split — fix raw data first.")
                return 1

            reports.append({"verify": verify_report})
            if verify_only:
                continue

            clean_report = self._clean_parquet_split(context)
            reports[-1]["clean"] = clean_report
            self._print_parquet_clean(clean_report)

        if not verify_only:
            self._write_anv_report(reports)

        print("\nVerification complete." if verify_only else "\nCleaning complete.")
        return 0

    def _verify_parquet_split(self, context: SplitContext) -> dict[str, Any]:
        issues: list[str] = []
        parquet_shards = [path.name for path in context.parquet_paths]
        rows = 0
        missing_audio = 0

        if not context.parquet_paths:
            issues.append("no_parquet_shards")
        else:
            sample_path = context.parquet_paths[0]
            schema_names = pq.read_schema(sample_path).names
            if not any(name in schema_names for name in ("audio", "bytes", "audio_bytes")):
                issues.append("missing_audio_column")
            text_col = resolve_parquet_text_column(schema_names)
            has_csv = context.transcripts_csv is not None or context.meta_csv is not None
            if text_col is None and not has_csv:
                issues.append("missing_transcript_source")

            audio_col = next(
                name for name in ("audio", "bytes", "audio_bytes") if name in schema_names
            )
            for parquet_path in context.parquet_paths[:3]:
                table = pq.read_table(parquet_path, columns=[audio_col])
                rows += table.num_rows
                for row_index in range(min(table.num_rows, 50)):
                    if not audio_bytes_from_struct(table.column(0)[row_index].as_py()):
                        missing_audio += 1

        return {
            "language": context.language,
            "split": context.split,
            "style": context.style,
            "folder": str(context.folder),
            "parquet_shards": parquet_shards,
            "rows": rows,
            "missing_audio": missing_audio,
            "issues": issues,
            "ok": not issues,
        }

    def _clean_parquet_split(self, context: SplitContext) -> dict[str, Any]:
        out_manifest = self.config.cleaned_manifest_path(context.split, context.style)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        meta_table = merge_meta_tables(context.meta_csv, context.transcripts_csv)
        shard_paths = [str(path) for path in context.parquet_paths]

        if self.config.workers > 1 and len(shard_paths) > 1:
            rows, stats = self._clean_parquet_parallel(context, shard_paths, meta_table)
        else:
            rows, stats = self._clean_parquet_sequential(context, shard_paths, meta_table)

        rows.sort(key=lambda row: (row["audio_source"]["path"], row["audio_source"]["row"]))
        with out_manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "language": context.language,
            "split": context.split,
            "style": context.style,
            "input_rows": stats.input_rows,
            "kept": stats.kept,
            "stats": stats.to_dict(),
            "output_manifest": str(out_manifest),
        }

    def _clean_parquet_sequential(
        self,
        context: SplitContext,
        shard_paths: list[str],
        meta_table: dict[str, dict[str, str]],
    ) -> tuple[list[dict[str, Any]], ParquetFilterStats]:
        all_rows: list[dict[str, Any]] = []
        total_stats = ParquetFilterStats()
        remaining = self.config.max_records

        for parquet_path in tqdm(shard_paths, desc=f"  {context.split}/{context.style}"):
            rows, stats = _process_parquet_shard(
                parquet_path,
                context.style,
                context.language,
                context.split,
                meta_table,
                self.config.min_duration_sec,
                self.config.max_duration_sec,
                self.config.verify_audio,
                remaining,
            )
            all_rows.extend(rows)
            self._merge_parquet_stats(total_stats, stats)
            if remaining is not None:
                remaining = max(0, remaining - len(rows))
                if remaining == 0:
                    break

        return all_rows, total_stats

    def _clean_parquet_parallel(
        self,
        context: SplitContext,
        shard_paths: list[str],
        meta_table: dict[str, dict[str, str]],
    ) -> tuple[list[dict[str, Any]], ParquetFilterStats]:
        workers = min(self.config.workers, len(shard_paths))
        print(f"  cleaning with {workers} worker(s), {len(shard_paths)} parquet shard(s)")

        all_rows: list[dict[str, Any]] = []
        total_stats = ParquetFilterStats()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_parquet_shard,
                    parquet_path,
                    context.style,
                    context.language,
                    context.split,
                    meta_table,
                    self.config.min_duration_sec,
                    self.config.max_duration_sec,
                    self.config.verify_audio,
                    self.config.max_records,
                ): parquet_path
                for parquet_path in shard_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {context.split}/{context.style}"):
                try:
                    rows, stats = future.result()
                except Exception as exc:
                    print(f"  ! shard failed ({futures[future]}): {exc}")
                    continue
                all_rows.extend(rows)
                self._merge_parquet_stats(total_stats, stats)

        if self.config.max_records is not None:
            all_rows = all_rows[: self.config.max_records]
        return all_rows, total_stats

    @staticmethod
    def _merge_parquet_stats(total: ParquetFilterStats, part: ParquetFilterStats) -> None:
        for field_name in total.to_dict():
            setattr(total, field_name, getattr(total, field_name) + getattr(part, field_name))

    def _write_anv_report(self, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.stats_dir / f"cleaning_report_{self.config.language}.json"
        path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved report: {path}")

    @staticmethod
    def _print_parquet_verify(report: dict[str, Any]) -> None:
        print(f"  parquet shards: {len(report['parquet_shards'])}")
        if report["parquet_shards"]:
            print(f"    e.g. {report['parquet_shards'][0]}")
        print(f"  sampled rows: {report['rows']}")
        print(f"  missing audio (sample): {report['missing_audio']}")
        if report["issues"]:
            print(f"  issues: {', '.join(report['issues'])}")

    @staticmethod
    def _print_parquet_clean(report: dict[str, Any]) -> None:
        stats = report["stats"]
        print(
            f"  kept: {report['kept']}/{report['input_rows']} "
            f"(empty={stats['empty_transcript']}, missing_audio={stats['missing_audio']}, "
            f"corrupt={stats['corrupt_audio']}, too_short={stats['too_short']}, "
            f"too_long={stats['too_long']}, dup={stats['duplicate_key']})"
        )
        print(f"  manifest: {report['output_manifest']}")


# Backward-compatible alias
AfrivoiceCleaningPipeline = CleaningPipeline
