#!/usr/bin/env python3
"""Smoke test for the Anv-ke Parquet pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.create_anv_fixture import build_fixture


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp) / "outputs"
        dataset_root = build_fixture(Path(tmp) / "anv_kalenjin")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "run_pipeline.py"),
            "--dataset-type",
            "anv",
            "--language",
            "kalenjin",
            "--dataset-root",
            str(dataset_root),
            "--work-dir",
            str(work_dir),
            "--split",
            "dev",
            "--style",
            "unscripted",
            "--skip-audio-check",
            "--workers",
            "2",
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

        manifest = work_dir / "cleaned" / "kalenjin" / "dev" / "unscripted" / "manifest_cleaned.jsonl"
        feat_dir = work_dir / "features" / "kalenjin" / "dev" / "unscripted"
        assert manifest.is_file(), f"missing manifest: {manifest}"

        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2, f"expected 2 cleaned rows, got {len(rows)}"
        assert rows[0]["audio_source"]["type"] == "parquet"

        npy_files = list(feat_dir.glob("*.npy"))
        assert len(npy_files) == 2, f"expected 2 feature files, got {len(npy_files)}"
        assert (feat_dir / "features.tsv").is_file()

        print("Anv-ke pipeline smoke test passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
