"""
Step 3 & 4: extract 80-bin log-mel features and apply SpecAugment masking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    FEATURES_DIR,
    FREQ_MASK_MAX_BINS,
    HOP_LENGTH,
    MASK_COUNT,
    N_FFT,
    N_MELS,
    PROCESSED_ROOT,
    SAMPLE_RATE,
    TIME_MASK_MAX_FRAMES,
    PipelineConfig,
)


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


class SpecAugmenter:
    """Apply time and frequency masking to log-mel features."""

    def __init__(
        self,
        time_mask_max: int = TIME_MASK_MAX_FRAMES,
        freq_mask_max: int = FREQ_MASK_MAX_BINS,
        mask_count: int = MASK_COUNT,
        seed: int = 42,
    ) -> None:
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.mask_count = mask_count
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def augment(self, mel: np.ndarray) -> np.ndarray:
        out = mel.copy()
        out = self._mask_axis(out, axis=1, max_width=self.time_mask_max)
        out = self._mask_axis(out, axis=0, max_width=self.freq_mask_max)
        return out

    def _mask_axis(self, mel: np.ndarray, axis: int, max_width: int) -> np.ndarray:
        out = mel.copy()
        size = out.shape[axis]
        fill_value = out.mean()
        for _ in range(self.mask_count):
            width = int(self._rng.integers(1, min(max_width, size) + 1))
            start = int(self._rng.integers(0, max(1, size - width)))
            slices = [slice(None), slice(None)]
            slices[axis] = slice(start, start + width)
            out[tuple(slices)] = fill_value
        return out


class AfrivoiceFeaturePipeline:
    """Extract and optionally augment log-mel features for all processed splits."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.extractor = LogMelFeatureExtractor()
        self.augmenter = SpecAugmenter()

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

    def run_augment(self, domain: str | None = None) -> int:
        reports: list[dict[str, Any]] = []
        for manifest_path in self._train_feature_manifests(domain):
            dom = manifest_path.parent.name
            print(f"\nAugmenting train features {dom}...")
            report = self._augment_train(dom, manifest_path)
            reports.append(report)
            print(f"  augmented: {report['augmented']}")

        if not reports:
            print("No train feature manifests found — skipping augmentation.")
            return 0

        self._write_report("augmentation_report.json", reports)
        print("\nAugmentation complete.")
        return 0

    def _extract_split(self, domain: str, split: str, manifest_path: Path) -> dict[str, Any]:
        feat_dir = FEATURES_DIR / domain / split
        feat_dir.mkdir(parents=True, exist_ok=True)
        out_manifest = FEATURES_DIR / domain / f"{split}_features.tsv"

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

    def _augment_train(self, domain: str, manifest_path: Path) -> dict[str, Any]:
        df = pd.read_csv(manifest_path, sep="\t")
        aug_dir = FEATURES_DIR / domain / "train_augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)
        rows = []

        for index, row in tqdm(df.iterrows(), total=len(df), desc="  augment"):
            mel = np.load(row["feature_path"])
            aug = self.augmenter.augment(mel)
            out_path = aug_dir / f"{Path(row['feature_path']).stem}_aug0.npy"
            np.save(out_path, aug)

            item = row.to_dict()
            item["feature_path"] = str(out_path.resolve())
            item["feature_shape"] = str(list(aug.shape))
            rows.append(item)

        out_manifest = FEATURES_DIR / domain / "train_augmented.tsv"
        pd.DataFrame(rows).to_csv(out_manifest, sep="\t", index=False)

        return {
            "domain": domain,
            "augmented": len(rows),
            "time_mask_max_frames": TIME_MASK_MAX_FRAMES,
            "freq_mask_max_bins": FREQ_MASK_MAX_BINS,
            "output_manifest": str(out_manifest),
        }

    def _targets(self, domain: str | None, split: str | None) -> list[tuple[str, str, Path]]:
        targets: list[tuple[str, str, Path]] = []
        if not PROCESSED_ROOT.is_dir():
            return targets

        for domain_dir in sorted(PROCESSED_ROOT.iterdir()):
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

    def _train_feature_manifests(self, domain: str | None) -> list[Path]:
        if not FEATURES_DIR.is_dir():
            return []
        manifests = []
        for domain_dir in sorted(FEATURES_DIR.iterdir()):
            if not domain_dir.is_dir():
                continue
            if domain and domain_dir.name != domain:
                continue
            manifest = domain_dir / "train_features.tsv"
            if manifest.is_file():
                manifests.append(manifest)
        return manifests

    def _write_report(self, filename: str, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.stats_dir / filename
        path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved report: {path}")
