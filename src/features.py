"""
Step 3: extract 80-bin log-mel features from preprocessed audio.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE, PipelineConfig


class LogMelFeatureExtractor:
    """Generate 80-bin log-mel spectrograms from 16 kHz WAV files."""

    def __init__(self) -> None:
        self.n_mels = N_MELS
        self.sample_rate = SAMPLE_RATE
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH

    def extract(self, wav_path: Path) -> np.ndarray:
        audio, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        return librosa.power_to_db(mel, ref=np.max)


class AfrivoiceFeaturePipeline:
    """Extract log-mel features for all processed splits."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.extractor = LogMelFeatureExtractor()

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
            print(f"  features: {report['extracted']}/{report['input']}")

        self._write_report("feature_extraction_report.json", reports)
        print("\nFeature extraction complete.")
        return 0

    def _extract_split(self, domain: str, split: str, manifest_path: Path) -> dict[str, Any]:
        feat_dir = self.config.features_dir / domain / split
        feat_dir.mkdir(parents=True, exist_ok=True)
        out_manifest = self.config.features_dir / domain / f"{split}_features.tsv"

        rows = []
        feature_paths = []
        feature_shapes = []

        with manifest_path.open(encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]

        for line in tqdm(lines, desc=f"  {domain}/{split}"):
            row = json.loads(line)
            wav = Path(row["audio_path"])
            if not wav.is_file():
                continue

            mel = self.extractor.extract(wav)
            out_path = feat_dir / f"{row.get('key', wav.stem)}.npy"
            np.save(out_path, mel)

            row["feature_path"] = str(out_path.resolve())
            row["feature_shape"] = str(list(mel.shape))
            rows.append(row)
            feature_paths.append(row["feature_path"])
            feature_shapes.append(row["feature_shape"])

            if self.config.max_records and len(rows) >= self.config.max_records:
                break

        df = pd.DataFrame(rows)
        if not df.empty:
            df["feature_path"] = feature_paths
            df["feature_shape"] = feature_shapes
            df.to_csv(out_manifest, sep="\t", index=False)

        return {
            "domain": domain,
            "split": split,
            "input": len(lines),
            "extracted": len(rows),
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

    def _write_report(self, filename: str, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.stats_dir / filename
        path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved report: {path}")
