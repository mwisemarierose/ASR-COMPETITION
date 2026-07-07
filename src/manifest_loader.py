"""
Stream manifest_*.jsonl files without loading the full dataset into memory.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .models import AfrivoiceRecord


class ManifestChunkLoader:
    """Read Afrivoice JSONL manifests one record at a time."""

    def __init__(self, manifest_paths: list[Path]) -> None:
        self.manifest_paths = manifest_paths

    def __iter__(self) -> Iterator[AfrivoiceRecord]:
        for manifest_path in self.manifest_paths:
            with manifest_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    yield AfrivoiceRecord(
                        raw=raw,
                        source_manifest=manifest_path,
                        line_number=line_number,
                    )

    def count_rows(self) -> int:
        total = 0
        for manifest_path in self.manifest_paths:
            with manifest_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        total += 1
        return total
