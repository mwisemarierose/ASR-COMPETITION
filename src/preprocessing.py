"""
Step 2: preprocess cleaned manifests to 16 kHz mono WAV.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm

from .audio_utils import load_audio_mono
from .config import DEFAULT_AUDIO_EXTENSION, SAMPLE_RATE, PipelineConfig

PreprocessResult = tuple[int, dict[str, Any] | None]


def _process_preprocess_row(row: dict[str, Any], out_audio_dir: Path) -> dict[str, Any] | None:
    """Convert one cleaned manifest row to a 16 kHz WAV file."""
    src = Path(row["audio_path"])
    if not src.is_file():
        return None

    dst = out_audio_dir / f"{row.get('key', src.stem)}.wav"
    try:
        audio = load_audio_mono(src, sample_rate=SAMPLE_RATE)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read audio {src}. For .webm files install ffmpeg: {exc}"
        ) from exc
    sf.write(dst, audio, SAMPLE_RATE)

    processed = dict(row)
    processed["source_audio_path"] = str(src.resolve())
    processed["source_audio_format"] = src.suffix.lower() or DEFAULT_AUDIO_EXTENSION
    processed["audio_path"] = str(dst.resolve())
    processed["sample_rate"] = SAMPLE_RATE
    return processed


def _process_preprocess_batch(
    batch: list[tuple[int, dict[str, Any]]],
    out_audio_dir: str,
) -> list[PreprocessResult]:
    """Process a batch of manifest rows in a worker process."""
    out_dir = Path(out_audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[PreprocessResult] = []
    for index, row in batch:
        processed = _process_preprocess_row(row, out_dir)
        results.append((index, processed))
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
            print(f"  processed: {report['processed']}/{report['input']} clips")

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
                "output_manifest": str(out_manifest),
            }

        if self.config.workers > 1:
            rows = self._process_split_parallel(domain, split, indexed_rows, out_audio_dir)
        else:
            rows = self._process_split_sequential(indexed_rows, out_audio_dir)

        self._write_manifest(out_manifest, rows)
        return {
            "domain": domain,
            "split": split,
            "input": len(indexed_rows),
            "processed": len(rows),
            "output_manifest": str(out_manifest),
        }

    def _process_split_sequential(
        self,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        out_audio_dir: Path,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, row in tqdm(indexed_rows, desc="  preprocess"):
            processed = _process_preprocess_row(row, out_audio_dir)
            if processed is not None:
                rows.append(processed)
        return rows

    def _process_split_parallel(
        self,
        domain: str,
        split: str,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        out_audio_dir: Path,
    ) -> list[dict[str, Any]]:
        workers = min(self.config.workers, len(indexed_rows))
        chunk_size = max(1, (len(indexed_rows) + workers - 1) // workers)
        batches = [
            indexed_rows[index : index + chunk_size]
            for index in range(0, len(indexed_rows), chunk_size)
        ]

        print(f"  preprocessing with {workers} worker(s), {len(batches)} batch(es)")
        batch_results: list[tuple[int, dict[str, Any] | None]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_preprocess_batch, batch, str(out_audio_dir))
                for batch in batches
            ]
            for future in tqdm(futures, desc=f"  {domain}/{split}"):
                batch_results.extend(future.result())

        batch_results.sort(key=lambda item: item[0])
        return [row for _, row in batch_results if row is not None]

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
