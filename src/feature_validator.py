"""
Step 5: validate generated log-mel features across all splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import N_MELS, PipelineConfig


class FeatureValidator:
    """Validate feature files have 80 mel bins and valid numeric values."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.expected_mels = N_MELS

    def run(self, domain: str | None = None, split: str | None = None) -> int:
        manifests = self._feature_manifests(domain, split)
        if not manifests:
            print("No feature manifests found. Run feature extraction first.")
            return 1

        reports: list[dict[str, Any]] = []
        all_ok = True

        for manifest_path in manifests:
            dom = manifest_path.parent.name
            spl = manifest_path.stem.replace("_features", "").replace("_augmented", "_augmented")
            if manifest_path.name == "train_augmented.tsv":
                spl = "train_augmented"
            else:
                spl = manifest_path.stem.replace("_features", "")

            print(f"\nValidating features {dom}/{spl}...")
            report = self._validate_manifest(manifest_path, dom, spl)
            reports.append(report)
            print(
                f"  checked: {report['checked']}  valid: {report['valid']}  "
                f"invalid: {report['invalid']}"
            )
            if not report["ok"]:
                all_ok = False

        self._write_report(reports)
        if all_ok:
            print("\nFeature validation complete — all checks passed.")
            return 0

        print("\nFeature validation failed for one or more splits.")
        return 1

    def _validate_manifest(self, manifest_path: Path, domain: str, split: str) -> dict[str, Any]:
        df = pd.read_csv(manifest_path, sep="\t")
        checked = 0
        valid = 0
        invalid = 0
        issues: list[str] = []

        for _, row in df.iterrows():
            checked += 1
            feat_path = Path(row["feature_path"])
            problems = self._check_file(feat_path)
            if problems:
                invalid += 1
                if len(issues) < 10:
                    issues.extend(f"{feat_path.name}: {problem}" for problem in problems)
            else:
                valid += 1

            if self.config.max_records and checked >= self.config.max_records:
                break

        return {
            "domain": domain,
            "split": split,
            "manifest": str(manifest_path),
            "checked": checked,
            "valid": valid,
            "invalid": invalid,
            "expected_mels": self.expected_mels,
            "issues": issues,
            "ok": invalid == 0 and valid > 0,
        }

    def _check_file(self, feat_path: Path) -> list[str]:
        problems: list[str] = []
        if not feat_path.is_file():
            return ["missing feature file"]

        try:
            mel = np.load(feat_path)
        except Exception as exc:
            return [f"cannot load array: {exc}"]

        if mel.ndim != 2:
            problems.append(f"expected 2D array, got shape {mel.shape}")
        elif mel.shape[0] != self.expected_mels:
            problems.append(f"expected {self.expected_mels} mel bins, got {mel.shape[0]}")

        if not np.isfinite(mel).all():
            problems.append("contains NaN or Inf values")

        return problems

    def _feature_manifests(self, domain: str | None, split: str | None) -> list[Path]:
        features_dir = self.config.features_dir
        if not features_dir.is_dir():
            return []

        manifests: list[Path] = []
        for domain_dir in sorted(features_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            if domain and domain_dir.name != domain:
                continue

            for manifest in sorted(domain_dir.glob("*_features.tsv")):
                spl = manifest.stem.replace("_features", "")
                if split and spl != split:
                    continue
                manifests.append(manifest)

            aug_manifest = domain_dir / "train_augmented.tsv"
            if aug_manifest.is_file() and (split is None or split == "train"):
                manifests.append(aug_manifest)

        return manifests

    def _write_report(self, reports: list[dict[str, Any]]) -> None:
        self.config.stats_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.stats_dir / "feature_validation_report.json"
        path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved report: {path}")
