"""
Step 3: extract 80-bin log-mel features from preprocessed audio.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE, PipelineConfig

ExtractResult = tuple[int, dict[str, Any] | None, str | None]

# Keep batches small so progress updates frequently on large train splits.
CLIPS_PER_BATCH = 250


def _wav_to_log_mel(wav_path: Path) -> np.ndarray:
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return librosa.power_to_db(mel, ref=np.max)


def _extract_feature_row(row: dict[str, Any], feat_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
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
        mel = _wav_to_log_mel(wav)
        np.save(out_path, mel)
    except Exception:
        return None, "corrupt_audio"

    processed = dict(row)
    processed["feature_path"] = str(out_path.resolve())
    processed["feature_shape"] = str(list(mel.shape))
    return processed, None


def _extract_feature_batch(
    batch: list[tuple[int, dict[str, Any]]],
    feat_dir: str,
) -> list[ExtractResult]:
    out_dir = Path(feat_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractResult] = []
    for index, row in batch:
        processed, skip_reason = _extract_feature_row(row, out_dir)
        results.append((index, processed, skip_reason))
    return results


class AfrivoiceFeaturePipeline:
    """Extract log-mel features for all processed splits."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run_extract(self, domain: str | None = None, split: str | None = None) -> int:
        targets = self._targets(domain, split)
        if not targets:
            print("No processed manifests found. Run preprocessing first.")
            return 1

        reports: list[dict[str, Any]] = []
        for dom, spl, manifest_path in targets:
            print(f"\nExtracting features {dom}/{spl}...")
            report = self._extract_split(dom, spl, manifest_path)
            reports.append(report)
            print(
                f"  features: {report['extracted']}/{report['input']} "
                f"(missing={report['missing_audio']}, corrupt={report['corrupt_audio']}, "
                f"resumed={report['resumed']})"
            )

        self._write_report(reports)
        print("\nFeature extraction complete.")
        return 0

    def _extract_split(self, domain: str, split: str, manifest_path: Path) -> dict[str, Any]:
        feat_dir = self.config.features_dir / domain / split
        feat_dir.mkdir(parents=True, exist_ok=True)
        out_manifest = self.config.features_dir / domain / f"{split}_features.tsv"

        indexed_rows = self._load_rows(manifest_path)
        if not indexed_rows:
            self._write_manifest(out_manifest, [])
            return self._empty_report(domain, split, out_manifest)

        if self.config.workers > 1:
            rows, stats = self._extract_split_parallel(domain, split, indexed_rows, feat_dir)
        else:
            rows, stats = self._extract_split_sequential(indexed_rows, feat_dir)

        self._write_manifest(out_manifest, rows)
        return {
            "domain": domain,
            "split": split,
            "input": len(indexed_rows),
            "extracted": len(rows),
            "missing_audio": stats["missing_audio"],
            "corrupt_audio": stats["corrupt_audio"],
            "resumed": stats["resumed"],
            "n_mels": N_MELS,
            "output_manifest": str(out_manifest),
        }

    def _extract_split_sequential(
        self,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        feat_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        batch_results: list[ExtractResult] = []
        for index, row in tqdm(indexed_rows, desc="  extract"):
            processed, skip_reason = _extract_feature_row(row, feat_dir)
            batch_results.append((index, processed, skip_reason))
        return self._collect_results(batch_results)

    def _extract_split_parallel(
        self,
        domain: str,
        split: str,
        indexed_rows: list[tuple[int, dict[str, Any]]],
        feat_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        workers = min(self.config.workers, len(indexed_rows))
        target_batches = max(workers * 8, (len(indexed_rows) + CLIPS_PER_BATCH - 1) // CLIPS_PER_BATCH)
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
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_size = {
                executor.submit(_extract_feature_batch, batch, str(feat_dir)): len(batch)
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
    def _empty_report(domain: str, split: str, out_manifest: Path) -> dict[str, Any]:
        return {
            "domain": domain,
            "split": split,
            "input": 0,
            "extracted": 0,
            "missing_audio": 0,
            "corrupt_audio": 0,
            "resumed": 0,
            "n_mels": N_MELS,
            "output_manifest": str(out_manifest),
        }

    def _targets(self, domain: str | None, split: str | None) -> list[tuple[str, str, Path]]:
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

    def _write_report(self, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for report in reports:
            by_domain.setdefault(report["domain"], []).append(report)

        saved_paths: list[Path] = []
        for dom, domain_reports in sorted(by_domain.items()):
            path = self.config.stats_dir / f"feature_extraction_report_{dom}.json"
            path.write_text(json.dumps(domain_reports, indent=2), encoding="utf-8")
            saved_paths.append(path)

        joined = ", ".join(str(path) for path in saved_paths)
        print(f"\nSaved report(s): {joined}")
