"""
Extract 80-bin log-mel features from preprocessed WAV or Parquet manifests.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .audio_utils import load_parquet_row_audio
from .config import (
    ANV_BATCH_TIMEOUT_SEC,
    ANV_CLIPS_PER_BATCH,
    ANV_EXTRACT_MAX_WORKERS,
    ANV_SHARD_CHUNK_SIZE,
    N_MELS,
    PipelineConfig,
)
from .mel_features import audio_to_log_mel, wav_path_to_log_mel

ExtractResult = tuple[int, dict[str, Any] | None, str | None]
CLIPS_PER_BATCH = 250
ANV_CLIP_TIMEOUT_SEC = 180


def _feature_out_path(row: dict[str, Any], feat_dir: Path) -> Path | None:
    audio_source = row.get("audio_source") or {}
    if audio_source.get("type") == "parquet":
        parquet_path = Path(audio_source["path"])
        row_index = int(audio_source["row"])
        return feat_dir / f"{row.get('key', f'{parquet_path.stem}_{row_index:06d}')}.npy"
    if "audio_path" in row:
        wav = Path(row["audio_path"])
        return feat_dir / f"{row.get('key', wav.stem)}.npy"
    return None


def _resumed_feature_result(index: int, row: dict[str, Any], out_path: Path) -> ExtractResult:
    processed = dict(row)
    processed["feature_path"] = str(out_path.resolve())
    if "feature_shape" not in processed:
        try:
            mel = np.load(out_path, mmap_mode="r")
            processed["feature_shape"] = str(list(mel.shape))
        except Exception:
            processed["feature_shape"] = row.get("feature_shape", "")
    return index, processed, "resumed"


def _extract_indexed_row_worker(index: int, row_json: str, feat_dir: str) -> ExtractResult:
    """Run one clip in a child process so native decode segfaults cannot kill the parent."""
    row = json.loads(row_json)
    processed, skip_reason = _extract_feature_row(row, Path(feat_dir))
    return index, processed, skip_reason


def _extract_indexed_row_isolated(
    index: int,
    row: dict[str, Any],
    feat_dir: Path,
    executor: ProcessPoolExecutor | None,
) -> tuple[ExtractResult, ProcessPoolExecutor | None]:
    row_json = json.dumps(row, ensure_ascii=False)
    feat_dir_str = str(feat_dir)
    retried = False

    while True:
        try:
            if executor is None:
                executor = ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)
            future = executor.submit(_extract_indexed_row_worker, index, row_json, feat_dir_str)
            return future.result(timeout=ANV_CLIP_TIMEOUT_SEC), executor
        except Exception:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            executor = None
            if retried:
                return (index, None, "corrupt_audio"), executor
            retried = True


def _serialize_indexed_rows(items: list[tuple[int, dict[str, Any]]]) -> str:
    return json.dumps([[index, row] for index, row in items], ensure_ascii=False)


def _deserialize_indexed_rows(payload: str) -> list[tuple[int, dict[str, Any]]]:
    return [(int(index), row) for index, row in json.loads(payload)]


def _extract_anv_work_unit(
    unit_kind: str,
    path: str,
    items_json: str,
    feat_dir: str,
) -> list[ExtractResult]:
    """Decode a Parquet shard chunk (or single row) inside one child process."""
    items = _deserialize_indexed_rows(items_json)
    out_dir = Path(feat_dir)
    if unit_kind == "parquet":
        return _extract_parquet_file_rows(Path(path), items, out_dir)

    results: list[ExtractResult] = []
    for index, row in items:
        processed, skip_reason = _extract_feature_row(row, out_dir)
        results.append((index, processed, skip_reason))
    return results


def _build_anv_work_units(
    pending: list[tuple[int, dict[str, Any]]],
    chunk_size: int,
) -> list[tuple[str, str, list[tuple[int, dict[str, Any]]]]]:
    units: list[tuple[str, str, list[tuple[int, dict[str, Any]]]]] = []
    by_path: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    for index, row in pending:
        audio_source = row.get("audio_source") or {}
        if audio_source.get("type") != "parquet":
            units.append(("row", "", [(index, row)]))
            continue
        by_path.setdefault(str(audio_source["path"]), []).append((index, row))

    for path, items in sorted(by_path.items()):
        for offset in range(0, len(items), chunk_size):
            units.append(("parquet", path, items[offset : offset + chunk_size]))
    return units


def _extract_wav_row(row: dict[str, Any], feat_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    wav = Path(row["audio_path"])
    if not wav.is_file():
        return None, "missing_audio"

    out_path = feat_dir / f"{row.get('key', wav.stem)}.npy"
    if out_path.is_file() and out_path.stat().st_size > 0:
        processed = dict(row)
        mel = np.load(out_path, mmap_mode="r")
        processed["feature_path"] = str(out_path.resolve())
        processed["feature_shape"] = str(list(mel.shape))
        return processed, "resumed"

    try:
        mel = wav_path_to_log_mel(wav)
        np.save(out_path, mel)
    except Exception:
        return None, "corrupt_audio"

    processed = dict(row)
    processed["feature_path"] = str(out_path.resolve())
    processed["feature_shape"] = str(list(mel.shape))
    return processed, None


def _extract_parquet_row(row: dict[str, Any], feat_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    audio_source = row.get("audio_source") or {}
    if audio_source.get("type") != "parquet":
        return None, "missing_audio"

    parquet_path = Path(audio_source["path"])
    row_index = int(audio_source["row"])
    out_path = feat_dir / f"{row.get('key', f'{parquet_path.stem}_{row_index:06d}')}.npy"

    if out_path.is_file() and out_path.stat().st_size > 0:
        processed = dict(row)
        mel = np.load(out_path, mmap_mode="r")
        processed["feature_path"] = str(out_path.resolve())
        processed["feature_shape"] = str(list(mel.shape))
        return processed, "resumed"

    if not parquet_path.is_file():
        return None, "missing_audio"

    try:
        audio = load_parquet_row_audio(parquet_path, row_index)
        mel = audio_to_log_mel(audio)
        np.save(out_path, mel)
    except Exception:
        return None, "corrupt_audio"

    processed = dict(row)
    processed["feature_path"] = str(out_path.resolve())
    processed["feature_shape"] = str(list(mel.shape))
    return processed, None


def _extract_feature_row(row: dict[str, Any], feat_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("audio_source", {}).get("type") == "parquet":
        return _extract_parquet_row(row, feat_dir)
    if "audio_path" in row:
        return _extract_wav_row(row, feat_dir)
    return None, "missing_audio"


def _extract_parquet_file_rows(
    parquet_path: Path,
    items: list[tuple[int, dict[str, Any]]],
    feat_dir: Path,
) -> list[ExtractResult]:
    """Process many manifest rows from one parquet shard with a single column read."""
    results: list[ExtractResult] = []
    pending: list[tuple[int, dict[str, Any], int, Path]] = []

    for index, row in items:
        row_index = int(row["audio_source"]["row"])
        out_path = feat_dir / f"{row.get('key', f'{parquet_path.stem}_{row_index:06d}')}.npy"

        if out_path.is_file() and out_path.stat().st_size > 0:
            processed = dict(row)
            processed["feature_path"] = str(out_path.resolve())
            if "feature_shape" not in processed:
                processed["feature_shape"] = row.get("feature_shape", "")
            results.append((index, processed, "resumed"))
            continue

        if not parquet_path.is_file():
            results.append((index, None, "missing_audio"))
            continue

        pending.append((index, row, row_index, out_path))

    if not pending:
        return results

    for index, row, row_index, out_path in pending:
        try:
            audio = load_parquet_row_audio(parquet_path, row_index)
            mel = audio_to_log_mel(audio)
            np.save(out_path, mel)
            processed = dict(row)
            processed["feature_path"] = str(out_path.resolve())
            processed["feature_shape"] = str(list(mel.shape))
            results.append((index, processed, None))
        except Exception:
            results.append((index, None, "corrupt_audio"))

    return results


def _extract_feature_batch(
    batch: list[tuple[int, dict[str, Any]]],
    feat_dir: str,
) -> list[ExtractResult]:
    out_dir = Path(feat_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractResult] = []

    parquet_by_path: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in batch:
        if row.get("audio_source", {}).get("type") == "parquet":
            path = str(row["audio_source"]["path"])
            parquet_by_path.setdefault(path, []).append((index, row))
            continue
        processed, skip_reason = _extract_feature_row(row, out_dir)
        results.append((index, processed, skip_reason))

    for path_str, items in parquet_by_path.items():
        results.extend(_extract_parquet_file_rows(Path(path_str), items, out_dir))

    return results


class FeaturePipeline:
    """Extract log-mel features for Afrivoice WAV or Anv-ke Parquet manifests."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run_extract(
        self,
        domain: str | None = None,
        split: str | None = None,
        style: str | None = None,
    ) -> int:
        if self.config.is_anv:
            return self._run_anv_extract(split=split, style=style)
        return self._run_afrivoice_extract(domain=domain, split=split)

    def _run_afrivoice_extract(self, domain: str | None, split: str | None) -> int:
        targets = self._afrivoice_targets(domain, split)
        if not targets:
            print("No processed manifests found. Run preprocessing first.")
            return 1

        reports: list[dict[str, Any]] = []
        for dom, spl, manifest_path in targets:
            print(f"\nExtracting features {dom}/{spl}...")
            report = self._extract_afrivoice_split(dom, spl, manifest_path)
            reports.append(report)
            print(
                f"  features: {report['extracted']}/{report['input']} "
                f"(missing={report['missing_audio']}, corrupt={report['corrupt_audio']}, "
                f"resumed={report['resumed']})"
            )

        self._write_afrivoice_report(reports)
        print("\nFeature extraction complete.")
        return 0

    def _run_anv_extract(self, split: str | None, style: str | None) -> int:
        targets = self._anv_targets(split, style)
        if not targets:
            print("No cleaned manifests found. Run clean first.")
            return 1

        reports: list[dict[str, Any]] = []
        for language, spl, sty, manifest_path in targets:
            print(f"\nExtracting features {language}/{spl}/{sty}...")
            report = self._extract_anv_split(language, spl, sty, manifest_path)
            reports.append(report)
            print(
                f"  features: {report['extracted']}/{report['input']} "
                f"(missing={report['missing_audio']}, corrupt={report['corrupt_audio']}, "
                f"resumed={report['resumed']})"
            )

        self._write_anv_report(reports)
        print("\nFeature extraction complete.")
        return 0

    def _extract_afrivoice_split(self, domain: str, split: str, manifest_path: Path) -> dict[str, Any]:
        feat_dir = self.config.features_dir / domain / split
        feat_dir.mkdir(parents=True, exist_ok=True)
        out_manifest = self.config.features_dir / domain / f"{split}_features.tsv"
        return self._extract_manifest(
            indexed_rows=self._load_rows(manifest_path),
            feat_dir=feat_dir,
            out_manifest=out_manifest,
            label=f"{domain}/{split}",
            report_key={"domain": domain, "split": split},
        )

    def _extract_anv_split(
        self,
        language: str,
        split: str,
        style: str,
        manifest_path: Path,
    ) -> dict[str, Any]:
        feat_dir = self.config.features_split_dir(split, style)
        feat_dir.mkdir(parents=True, exist_ok=True)
        out_manifest = feat_dir / "features.tsv"
        return self._extract_manifest(
            indexed_rows=self._load_rows(manifest_path),
            feat_dir=feat_dir,
            out_manifest=out_manifest,
            label=f"{language}/{split}/{style}",
            report_key={"language": language, "split": split, "style": style},
        )

    def _extract_manifest(
        self,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        feat_dir: Path,
        out_manifest: Path,
        label: str,
        report_key: dict[str, str],
    ) -> dict[str, Any]:
        if not indexed_rows:
            self._write_manifest(out_manifest, [])
            return {**report_key, **self._empty_counts(out_manifest)}

        if self.config.is_anv:
            rows, stats = self._extract_anv_safe(label, indexed_rows, feat_dir)
        elif self.config.workers > 1:
            rows, stats = self._extract_parallel(label, indexed_rows, feat_dir)
        else:
            rows, stats = self._extract_sequential(indexed_rows, feat_dir)

        self._write_manifest(out_manifest, rows)
        return {
            **report_key,
            "input": len(indexed_rows),
            "extracted": len(rows),
            "missing_audio": stats["missing_audio"],
            "corrupt_audio": stats["corrupt_audio"],
            "resumed": stats["resumed"],
            "n_mels": N_MELS,
            "output_manifest": str(out_manifest),
        }

    def _extract_anv_safe(
        self,
        label: str,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        feat_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Anv-ke extract: resume in parent; decode Parquet shard chunks in child processes."""
        feat_dir.mkdir(parents=True, exist_ok=True)
        workers = min(max(1, self.config.workers), ANV_EXTRACT_MAX_WORKERS, len(indexed_rows))

        batch_results: list[ExtractResult] = []
        pending: list[tuple[int, dict[str, Any]]] = []
        for index, row in indexed_rows:
            out_path = _feature_out_path(row, feat_dir)
            if out_path is not None and out_path.is_file() and out_path.stat().st_size > 0:
                batch_results.append(_resumed_feature_result(index, row, out_path))
            else:
                pending.append((index, row))

        print(
            f"  extracting {len(pending)} clip(s) with {workers} worker(s), "
            f"shard chunk size {ANV_SHARD_CHUNK_SIZE} "
            f"({len(batch_results)} resumed)"
        )

        work_units = _build_anv_work_units(pending, ANV_SHARD_CHUNK_SIZE)
        failed_units: list[tuple[str, str, list[tuple[int, dict[str, Any]]]]] = []
        logged_errors = 0
        feat_dir_str = str(feat_dir)

        if work_units:
            with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
                future_to_unit = {
                    executor.submit(
                        _extract_anv_work_unit,
                        unit_kind,
                        path,
                        _serialize_indexed_rows(chunk),
                        feat_dir_str,
                    ): (unit_kind, path, chunk)
                    for unit_kind, path, chunk in work_units
                }
                with tqdm(total=len(indexed_rows), initial=len(batch_results), desc=f"  {label}") as progress:
                    for future in as_completed(future_to_unit):
                        unit_kind, path, chunk = future_to_unit[future]
                        try:
                            batch_results.extend(future.result(timeout=ANV_BATCH_TIMEOUT_SEC))
                        except Exception as exc:
                            if logged_errors < 5:
                                print(f"  ! shard chunk failed ({len(chunk)} clips): {exc}")
                                logged_errors += 1
                            if len(chunk) > 1:
                                mid = len(chunk) // 2
                                failed_units.append((unit_kind, path, chunk[:mid]))
                                failed_units.append((unit_kind, path, chunk[mid:]))
                            else:
                                failed_units.append((unit_kind, path, chunk))
                        progress.update(len(chunk))

        while failed_units:
            print(f"  retrying {sum(len(chunk) for _, _, chunk in failed_units)} clip(s) in smaller chunks...")
            retry_units = failed_units
            failed_units = []
            with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
                future_to_unit = {
                    executor.submit(
                        _extract_anv_work_unit,
                        unit_kind,
                        path,
                        _serialize_indexed_rows(chunk),
                        feat_dir_str,
                    ): (unit_kind, path, chunk)
                    for unit_kind, path, chunk in retry_units
                }
                with tqdm(total=len(retry_units), desc="  retry", leave=False) as progress:
                    for future in as_completed(future_to_unit):
                        unit_kind, path, chunk = future_to_unit[future]
                        try:
                            batch_results.extend(future.result(timeout=ANV_BATCH_TIMEOUT_SEC))
                        except Exception:
                            executor: ProcessPoolExecutor | None = None
                            for index, row in chunk:
                                result, executor = _extract_indexed_row_isolated(
                                    index, row, feat_dir, executor
                                )
                                batch_results.append(result)
                            if executor is not None:
                                executor.shutdown(wait=True)
                        progress.update(len(chunk))

        return self._collect_results(batch_results)

    def _extract_sequential(
        self,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        feat_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        batch_results: list[ExtractResult] = []
        for index, row in tqdm(indexed_rows, desc="  extract"):
            processed, skip_reason = _extract_feature_row(row, feat_dir)
            batch_results.append((index, processed, skip_reason))
        return self._collect_results(batch_results)

    def _extract_parallel(
        self,
        label: str,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        feat_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        workers = min(self.config.workers, len(indexed_rows))
        clips_per_batch = ANV_CLIPS_PER_BATCH if self.config.is_anv else CLIPS_PER_BATCH
        target_batches = max(workers * 8, (len(indexed_rows) + clips_per_batch - 1) // clips_per_batch)
        chunk_size = max(1, (len(indexed_rows) + target_batches - 1) // target_batches)
        batches = [
            indexed_rows[index : index + chunk_size]
            for index in range(0, len(indexed_rows), chunk_size)
        ]

        print(
            f"  extracting with {workers} worker(s), "
            f"{len(indexed_rows)} clips, ~{chunk_size} clips/batch"
        )
        batch_results: list[ExtractResult] = []
        failed_batches: list[list[tuple[int, dict[str, Any]]]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_batch = {
                executor.submit(_extract_feature_batch, batch, str(feat_dir)): batch
                for batch in batches
            }
            with tqdm(total=len(indexed_rows), desc=f"  {label}") as progress:
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results.extend(future.result())
                    except Exception as exc:
                        print(f"  ! worker batch failed: {exc}")
                        failed_batches.append(batch)
                    progress.update(len(batch))

        if failed_batches:
            print(f"  retrying {len(failed_batches)} failed batch(es) sequentially...")
            for batch in failed_batches:
                batch_results.extend(_extract_feature_batch(batch, str(feat_dir)))

        return self._collect_results(batch_results)

    @staticmethod
    def _collect_results(
        batch_results: list[ExtractResult],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        stats = {"missing_audio": 0, "corrupt_audio": 0, "resumed": 0}
        rows: list[dict[str, Any]] = []

        for _, processed, skip_reason in sorted(batch_results, key=lambda item: item[0]):
            if processed is not None:
                rows.append(processed)
                if skip_reason == "resumed":
                    stats["resumed"] += 1
                continue
            if skip_reason == "missing_audio":
                stats["missing_audio"] += 1
            elif skip_reason == "corrupt_audio":
                stats["corrupt_audio"] += 1

        return rows, stats

    def _load_rows(self, manifest_path: Path) -> list[tuple[int, dict[str, Any]]]:
        rows: list[tuple[int, dict[str, Any]]] = []
        with manifest_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                rows.append((index, json.loads(line)))
                if self.config.max_records and len(rows) >= self.config.max_records:
                    break
        return rows

    @staticmethod
    def _write_manifest(out_manifest: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            if out_manifest.is_file():
                out_manifest.unlink()
            return
        df = pd.DataFrame(rows)
        df.to_csv(out_manifest, sep="\t", index=False)

    @staticmethod
    def _empty_counts(out_manifest: Path) -> dict[str, Any]:
        return {
            "input": 0,
            "extracted": 0,
            "missing_audio": 0,
            "corrupt_audio": 0,
            "resumed": 0,
            "n_mels": N_MELS,
            "output_manifest": str(out_manifest),
        }

    def _afrivoice_targets(self, domain: str | None, split: str | None) -> list[tuple[str, str, Path]]:
        targets: list[tuple[str, str, Path]] = []
        processed_root = self.config.processed_root
        if not processed_root.is_dir():
            return targets

        for domain_dir in sorted(processed_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            dom = domain_dir.name
            if domain and dom != domain:
                continue
            for split_dir in sorted(domain_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                spl = split_dir.name
                if split and spl != split:
                    continue
                manifest = split_dir / "manifest_processed.jsonl"
                if manifest.is_file():
                    targets.append((dom, spl, manifest))
        return targets

    def _anv_targets(
        self,
        split: str | None,
        style: str | None,
    ) -> list[tuple[str, str, str, Path]]:
        targets: list[tuple[str, str, str, Path]] = []
        if not self.config.language:
            return targets

        cleaned_root = self.config.output_root / self.config.language
        if not cleaned_root.is_dir():
            return targets

        for split_dir in sorted(cleaned_root.iterdir()):
            if not split_dir.is_dir():
                continue
            spl = split_dir.name
            if split and spl != split:
                continue
            for style_dir in sorted(split_dir.iterdir()):
                if not style_dir.is_dir():
                    continue
                sty = style_dir.name
                if style and sty != style:
                    continue
                manifest = style_dir / "manifest_cleaned.jsonl"
                if manifest.is_file():
                    targets.append((self.config.language, spl, sty, manifest))
        return targets

    def _write_afrivoice_report(self, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for report in reports:
            by_domain.setdefault(report["domain"], []).append(report)

        saved_paths: list[Path] = []
        for dom, domain_reports in sorted(by_domain.items()):
            path = self.config.stats_dir / f"feature_extraction_report_{dom}.json"
            path.write_text(json.dumps(domain_reports, indent=2), encoding="utf-8")
            saved_paths.append(path)

        print(f"\nSaved report(s): {', '.join(str(path) for path in saved_paths)}")

    def _write_anv_report(self, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.stats_dir / f"feature_extraction_report_{self.config.language}.json"
        path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved report: {path}")


# Backward-compatible alias
AfrivoiceFeaturePipeline = FeaturePipeline
