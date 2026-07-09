#!/usr/bin/env python3
"""Smoke test for Afrivoice Somali flat-folder layout."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import PipelineConfig
from src.discovery import DatasetDiscovery
from tests.create_somali_fixture import build_fixture


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        dataset_root = build_fixture(Path(tmp) / "Afrivoice")
        config = PipelineConfig(
            dataset_type="afrivoice",
            language="somali",
            dataset_root=dataset_root,
            verify_audio=False,
            skip_verify=True,
        )
        targets = list(DatasetDiscovery(config).iter_targets())
        assert len(targets) == 1, f"expected 1 target, got {len(targets)}"
        context = targets[0]
        assert context.domain == "somali"
        assert context.split == "all"
        assert context.audio_dir.name == "audio_shards"
        assert context.manifest_paths[0].name == "manifest_0.json"

        work_dir = Path(tmp) / "outputs"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "run_pipeline.py"),
            "--dataset-type",
            "afrivoice",
            "--language",
            "somali",
            "--dataset-root",
            str(dataset_root),
            "--work-dir",
            str(work_dir),
            "--step",
            "clean",
            "--skip-audio-check",
            "--skip-verify",
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

        manifest = work_dir / "cleaned" / "somali" / "all" / "manifest_cleaned.jsonl"
        assert manifest.is_file(), f"missing manifest: {manifest}"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2, f"expected 2 cleaned rows, got {len(rows)}"

    print("Afrivoice Somali flat-layout smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
