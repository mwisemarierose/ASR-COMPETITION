"""
Step 2: preprocess cleaned manifests to 16 kHz mono WAV.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm

from .audio_utils import load_audio_mono
from .config import DEFAULT_AUDIO_EXTENSION, SAMPLE_RATE, PipelineConfig

PreprocessResult = tuple[int, dict[str, Any] | None, str | None]

# Keep batches small so progress updates frequently on large train splits.
CLIPS_PER_BATCH = 250


def _build_processed_row(row: dict[str, Any], src: Path, dst: Path) -> dict[str, Any]:
    processed = dict(row)
    processed["source_audio_path"] = str(src.resolve())
    processed["source_audio_format"] = src.suffix.lower() or DEFAULT_AUDIO_EXTENSION
    processed["audio_path"] = str(dst.resolve())
    processed["sample_rate"] = SAMPLE_RATE
    return processed


def _process_preprocess_row(row: dict[str, Any], out_audio_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one cleaned manifest row to a 16 kHz WAV file."""
    src = Path(row["audio_path"])
    if not src.is_file():
        return None, "missing_audio"

    dst = out_audio_dir / f"{row.get('key', src.stem)}.wav"
    if dst.is_file() and dst.stat().st_size > 0:
        return _build_processed_row(row, src, dst), "resumed"

    try:
        audio = load_audio_mono(src, sample_rate=SAMPLE_RATE)
        sf.write(dst, audio, SAMPLE_RATE)
    except Exception:
        return None, "corrupt_audio"

    return _build_processed_row(row, src, dst), None


def _process_preprocess_batch(
    batch: list[tuple[int, dict[str, Any]]],
    out_audio_dir: str,
) -> list[PreprocessResult]:
    """Process a batch of manifest rows in a worker process."""
    out_dir = Path(out_audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[PreprocessResult] = []
    for index, row in batch:
        processed, skip_reason = _process_preprocess_row(row, out_dir)
        results.append((index, processed, skip_reason))
    return results


class AfrivoicePreprocessingPipeline:
    """Resample cleaned audio to 16 kHz and write processed manifests."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.processed_root = config.processed_root

    def run(self, domain: str | None = None, split: str | None = None) -> int:
        targets = self._targets(domain, split)
        if not targets:
            print("No cleaned manifests found. Run cleaning first.")
            return 1

        reports: list[dict[str, Any]] = []
        for dom, spl, manifest_path in targets:
            print(f"\nPreprocessing {dom}/{spl}...")
            report = self._process_split(dom, spl, manifest_path)
            reports.append(report)
            print(
                f"  processed: {report['processed']}/{report['input']} clips "
                f"(missing={report['missing_audio']}, corrupt={report['corrupt_audio']}, "
                f"resumed={report['resumed']})"
            )

        self._write_report(reports)
        print("\nPreprocessing complete.")
        return 0

    def _process_split(self, domain: str, split: str, manifest_path: Path) -> dict[str, Any]:
        out_audio_dir = self.processed_root / domain / split / "audio"
        out_manifest = self.processed_root / domain / split / "manifest_processed.jsonl"
        out_audio_dir.mkdir(parents=True, exist_ok=True)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)

        indexed_rows = self._load_rows(manifest_path)
        if not indexed_rows:
            self._write_manifest(out_manifest, [])
            return {
                "domain": domain,
                "split": split,
                "input": 0,
                "processed": 0,
                "missing_audio": 0,
                "corrupt_audio": 0,
                "resumed": 0,
                "output_manifest": str(out_manifest),
            }

        if self.config.workers > 1:
            rows, stats = self._process_split_parallel(domain, split, indexed_rows, out_audio_dir)
        else:
            rows, stats = self._process_split_sequential(indexed_rows, out_audio_dir)

        self._write_manifest(out_manifest, rows)
        return {
            "domain": domain,
            "split": split,
            "input": len(indexed_rows),
            "processed": len(rows),
            "missing_audio": stats["missing_audio"],
            "corrupt_audio": stats["corrupt_audio"],
            "resumed": stats["resumed"],
            "output_manifest": str(out_manifest),
        }

    @staticmethod
    def _collect_results(
        batch_results: list[PreprocessResult],
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

    def _process_split_sequential(
        self,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        out_audio_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        batch_results: list[PreprocessResult] = []
        for index, row in tqdm(indexed_rows, desc="  preprocess"):
            processed, skip_reason = _process_preprocess_row(row, out_audio_dir)
            batch_results.append((index, processed, skip_reason))
        return self._collect_results(batch_results)

    def _process_split_parallel(
        self,
        domain: str,
        split: str,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        out_audio_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        workers = min(self.config.workers, len(indexed_rows))
        target_batches = max(workers * 8, (len(indexed_rows) + CLIPS_PER_BATCH - 1) // CLIPS_PER_BATCH)
        chunk_size = max(1, (len(indexed_rows) + target_batches - 1) // target_batches)
        batches = [
            indexed_rows[index : index + chunk_size]
            for index in range(0, len(indexed_rows), chunk_size)
        ]

        print(
            f"  preprocessing with {workers} worker(s), "
            f"{len(indexed_rows)} clips, ~{chunk_size} clips/batch"
        )
        batch_results: list[PreprocessResult] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_size = {
                executor.submit(_process_preprocess_batch, batch, str(out_audio_dir)): len(batch)
                for batch in batches
            }
            with tqdm(total=len(indexed_rows), desc=f"  {domain}/{split}") as progress:
                for future in as_completed(future_to_size):
                    try:
                        batch_results.extend(future.result())
                    except Exception as exc:
                        print(f"  ! worker batch failed: {exc}")
                    progress.update(future_to_size[future])

        return self._collect_results(batch_results)

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
        with out_manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _targets(self, domain: str | None, split: str | None) -> list[tuple[str, str, Path]]:
        root = self.config.output_root
        if not root.is_dir():
            return []

        targets: list[tuple[str, str, Path]] = []
        for domain_dir in sorted(root.iterdir()):
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
                manifest = split_dir / "manifest_cleaned.jsonl"
                if manifest.is_file():
                    targets.append((dom, spl, manifest))
        return targets

    def _write_report(self, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for report in reports:
            by_domain.setdefault(report["domain"], []).append(report)

        saved_paths: list[Path] = []
        for dom, domain_reports in sorted(by_domain.items()):
            path = self.config.stats_dir / f"preprocessing_report_{dom}.json"
            path.write_text(json.dumps(domain_reports, indent=2), encoding="utf-8")
            saved_paths.append(path)

        joined = ", ".join(str(path) for path in saved_paths)
        print(f"\nSaved report(s): {joined}")
