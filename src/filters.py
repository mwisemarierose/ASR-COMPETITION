"""
Apply validators and transcript cleaning to Afrivoice records.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_AUDIO_EXTENSION, PipelineConfig
from .media_resolver import MediaResolver
from .models import AfrivoiceRecord, FilterStats, SplitContext
from .transcript_cleaner import SwahiliTranscriptCleaner
from .validators import (
    AlignmentValidator,
    AudioDurationValidator,
    AudioFileValidator,
    ImageFileValidator,
    TranscriptValidator,
)


class RecordFilter:
    """Run validation chain and produce cleaned output rows."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.transcript_cleaner = SwahiliTranscriptCleaner()
        self.validators = [
            TranscriptValidator(),
            AudioFileValidator(),
        ]
        if config.verify_images:
            self.validators.append(ImageFileValidator())
        self.validators.append(
            AudioDurationValidator(
                min_duration_sec=config.min_duration_sec,
                max_duration_sec=config.max_duration_sec,
                verify_audio=config.verify_audio,
            )
        )
        if config.verify_alignment and config.verify_audio:
            self.validators.append(AlignmentValidator())
        self._seen_keys: set[str] = set()

    def process(self, record: AfrivoiceRecord, context: SplitContext) -> tuple[dict[str, Any] | None, str | None]:
        for validator in self.validators:
            ok, reason = validator.validate(record, context)
            if not ok:
                return None, reason

        if record.key:
            if record.key in self._seen_keys:
                return None, "duplicate_key"
            self._seen_keys.add(record.key)

        cleaned = dict(record.raw)
        cleaned["transcript"] = self.transcript_cleaner.clean(record)
        cleaned["audio_path"] = str(MediaResolver.resolve_audio_record(context, record).resolve())
        cleaned["source_audio_format"] = Path(cleaned["audio_path"]).suffix.lower() or DEFAULT_AUDIO_EXTENSION
        cleaned["duration_sec"] = self._duration(record, context)
        return cleaned, None

    def _duration(self, record: AfrivoiceRecord, context: SplitContext) -> float | None:
        validator = self.validators[-1]
        if isinstance(validator, AudioDurationValidator):
            return validator._duration(record, context)
        return record.manifest_duration

    def bump_stat(self, stats: FilterStats, reason: str | None) -> None:
        if reason is None:
            stats.kept += 1
            return
        if hasattr(stats, reason):
            setattr(stats, reason, getattr(stats, reason) + 1)
