"""
Stream manifest JSONL files and load Anv-ke CSV metadata.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .models import AfrivoiceRecord


def _iter_manifest_objects(manifest_path: Path):
    """Yield (raw_dict, line_number) from JSONL or JSON-array manifest files."""
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return

    if text.startswith("["):
        for line_number, raw in enumerate(json.loads(text), start=1):
            if isinstance(raw, dict):
                yield line_number, raw
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if isinstance(raw, dict):
            yield line_number, raw


class ManifestChunkLoader:
    """Read Afrivoice JSONL manifests one record at a time."""

    def __init__(self, manifest_paths: list[Path]) -> None:
        self.manifest_paths = manifest_paths

    def __iter__(self) -> Iterator[AfrivoiceRecord]:
        for manifest_path in self.manifest_paths:
            for line_number, raw in _iter_manifest_objects(manifest_path):
                yield AfrivoiceRecord(
                    raw=raw,
                    source_manifest=manifest_path,
                    line_number=line_number,
                )

    def count_rows(self) -> int:
        total = 0
        for manifest_path in self.manifest_paths:
            for _line_number, _raw in _iter_manifest_objects(manifest_path):
                total += 1
        return total


def _normalize_csv_key(name: str | None) -> str:
    if not name:
        return ""
    return name.strip().lower().replace(" ", "_")


def _csv_cell_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _csv_row_lookup_key(row: dict[str, str]) -> str | None:
    for field in ("recorder_uuid", "mediaPathId", "media_path_id", "id"):
        value = row.get(field) or row.get(_normalize_csv_key(field))
        if value:
            return str(value).strip()
    return None


def load_csv_table(path: Path) -> dict[str, dict[str, str]]:
    """Load meta.csv or transcripts.csv keyed by recorder_uuid / mediaPathId."""
    if not path.is_file():
        return {}

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}

        table: dict[str, dict[str, str]] = {}
        for raw_row in reader:
            row: dict[str, str] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                norm_key = _normalize_csv_key(key)
                if not norm_key:
                    continue
                row[norm_key] = _csv_cell_value(value)
            lookup = _csv_row_lookup_key(row)
            if lookup:
                table[lookup] = row
        return table


def merge_meta_tables(
    meta_csv: Path | None,
    transcripts_csv: Path | None,
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for path in (meta_csv, transcripts_csv):
        if not path:
            continue
        for key, row in load_csv_table(path).items():
            merged.setdefault(key, {}).update(row)
    return merged


def transcript_from_csv_row(row: dict[str, str], style: str) -> str:
    if style == "scripted":
        fields = ("actualsentence", "actual_sentence", "sentence", "transcription", "transcript", "text")
    else:
        fields = ("transcript", "transcription", "text", "sentence")

    for field in fields:
        value = row.get(field, "")
        if value:
            return value

    for field in ("transcription", "transcript", "text", "actualsentence", "actual_sentence"):
        value = row.get(field, "")
        if value:
            return value
    return ""


def enrich_csv_row(meta: dict[str, str] | None, style: str) -> dict[str, Any]:
    if not meta:
        return {}

    enriched: dict[str, Any] = {}
    for src_field, dst_field in (
        ("recorder_uuid", "recorder_uuid"),
        ("domain", "domain"),
        ("dialect", "dialect"),
        ("sentencedialect", "dialect"),
        ("language", "language"),
        ("type", "utterance_type"),
    ):
        value = meta.get(src_field, "")
        if value and dst_field not in enriched:
            enriched[dst_field] = value

    transcript = transcript_from_csv_row(meta, style)
    if transcript:
        enriched["csv_transcript"] = transcript

    duration = meta.get("duration", "")
    if duration:
        try:
            enriched["csv_duration_sec"] = float(duration)
        except (TypeError, ValueError):
            pass

    return enriched
