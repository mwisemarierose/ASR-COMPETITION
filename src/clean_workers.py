"""
Worker helpers for parallel manifest cleaning.
"""
from __future__ import annotations

from typing import Any

from .config import PipelineConfig
from .filters import RecordFilter
from .models import AfrivoiceRecord, SplitContext

SortKey = tuple[str, int]
CleanResult = tuple[SortKey, dict[str, Any] | None, str | None]


def process_clean_batch(
    batch: list[AfrivoiceRecord],
    config: PipelineConfig,
    context: SplitContext,
) -> list[CleanResult]:
    """Filter and normalize a batch of manifest rows (no duplicate-key dedup)."""
    record_filter = RecordFilter(config)
    results: list[CleanResult] = []
    for record in batch:
        cleaned, reason = record_filter.process(record, context)
        sort_key = (str(record.source_manifest), record.line_number)
        results.append((sort_key, cleaned, reason))
    return results
