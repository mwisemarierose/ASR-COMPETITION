#!/usr/bin/env python3
"""Tests for Anv-ke dataset path resolution."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import PipelineConfig


def _write_anv_layout(root: Path) -> None:
    (root / "dev" / "unscripted" / "audios").mkdir(parents=True, exist_ok=True)


def test_resolves_capitalized_language_folder_from_parent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        datasets = Path(tmp) / "datasets"
        _write_anv_layout(datasets / "Maasai")

        config = PipelineConfig(
            dataset_type="anv",
            language="maasai",
            dataset_root=datasets,
        )
        assert config.language_dataset_root().resolve() == (datasets / "Maasai").resolve()


def test_resolve_anv_language_folder_finds_canonical_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        datasets = Path(tmp) / "datasets"
        _write_anv_layout(datasets / "Maasai")

        config = PipelineConfig(
            dataset_type="anv",
            language="maasai",
            dataset_root=datasets,
        )
        resolved = config._resolve_anv_language_folder(datasets)
        assert resolved is not None
        assert resolved.resolve() == (datasets / "Maasai").resolve()


def test_resolve_anv_language_folder_finds_case_insensitive_match() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        datasets = Path(tmp) / "datasets"
        _write_anv_layout(datasets / "Maasai")

        config = PipelineConfig(
            dataset_type="anv",
            language="maasai",
            dataset_root=datasets,
        )
        resolved = config._resolve_anv_language_folder(datasets)
        assert resolved is not None
        assert resolved.name == "Maasai"


def test_direct_language_root_still_works() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "Kalenjin"
        _write_anv_layout(root)
        config = PipelineConfig(
            dataset_type="anv",
            language="kalenjin",
            dataset_root=root,
        )
        assert config.language_dataset_root().resolve() == root.resolve()


if __name__ == "__main__":
    test_resolves_capitalized_language_folder_from_parent()
    test_resolve_anv_language_folder_finds_canonical_name()
    test_resolve_anv_language_folder_finds_case_insensitive_match()
    test_direct_language_root_still_works()
    print("Anv path resolution tests passed.")
