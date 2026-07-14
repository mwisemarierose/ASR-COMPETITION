"""Tests for manifest JSONL loading edge cases."""
from __future__ import annotations

from pathlib import Path

from src.manifest_loader import ManifestChunkLoader, _iter_manifest_objects


def test_skips_blank_and_invalid_jsonl_lines(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "manifest_0.jsonl"
    manifest.write_text(
        '{"key": "a", "transcription": "habari"}\n'
        "\n"
        "\u200b\n"
        "{not valid json}\n"
        '{"key": "b", "transcription": "asante"}\n',
        encoding="utf-8",
    )

    rows = list(_iter_manifest_objects(manifest))
    assert len(rows) == 2
    assert rows[0][1]["key"] == "a"
    assert rows[1][1]["key"] == "b"

    captured = capsys.readouterr()
    assert "skipping invalid JSON" in captured.out


def test_manifest_chunk_loader_counts_valid_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest_0.jsonl"
    manifest.write_text('{"transcription": "one"}\n{"transcription": "two"}\n', encoding="utf-8")
    loader = ManifestChunkLoader([manifest])
    assert loader.count_rows() == 2
    assert len(list(loader)) == 2
