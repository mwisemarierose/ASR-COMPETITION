"""
Step 2: preprocess cleaned manifests to 16 kHz mono WAV.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
from tqdm import tqdm

from .config import DEFAULT_AUDIO_EXTENSION, SAMPLE_RATE, PipelineConfig


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

        processed = 0
        rows: list[dict[str, Any]] = []

        with manifest_path.open(encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]

        for line in tqdm(lines, desc=f"  {domain}/{split}"):
            row = json.loads(line)
            src = Path(row["audio_path"])
            if not src.is_file():
                continue

            dst = out_audio_dir / f"{row.get('key', src.stem)}.wav"
            try:
                audio, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot read audio {src}. For .webm files install ffmpeg: {exc}"
                ) from exc
            sf.write(dst, audio, SAMPLE_RATE)

            row["source_audio_path"] = str(src.resolve())
            row["source_audio_format"] = src.suffix.lower() or DEFAULT_AUDIO_EXTENSION

            row["audio_path"] = str(dst.resolve())
            row["sample_rate"] = SAMPLE_RATE
            rows.append(row)
            processed += 1

            if self.config.max_records and processed >= self.config.max_records:
                break

        with out_manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "domain": domain,
            "split": split,
            "input": len(lines),
            "processed": processed,
            "output_manifest": str(out_manifest),
        }

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
        path = self.config.stats_dir / "preprocessing_report.json"
        path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved report: {path}")
