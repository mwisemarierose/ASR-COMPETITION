"""
Validators for Afrivoice ASR records and audio files.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .audio_utils import get_audio_duration
from .config import (
    ALIGNMENT_DURATION_TOLERANCE_SEC,
    MAX_CHARS_PER_SEC,
    MIN_CHARS_PER_SEC,
)
from .media_resolver import MediaResolver
from .models import AfrivoiceRecord, SplitContext


class BaseValidator(ABC):
    """Shared validator interface."""

    @abstractmethod
    def validate(self, record: AfrivoiceRecord, context: SplitContext) -> tuple[bool, str | None]:
        """Return (is_valid, rejection_reason)."""


class TranscriptValidator(BaseValidator):
    """Reject records with empty transcription text."""

    def validate(self, record: AfrivoiceRecord, context: SplitContext) -> tuple[bool, str | None]:
        if record.normalized_transcription or record.transcription:
            return True, None
        return False, "empty_transcript"


class AudioFileValidator(BaseValidator):
    """Reject records whose audio file is missing on disk."""

    def validate(self, record: AfrivoiceRecord, context: SplitContext) -> tuple[bool, str | None]:
        if not record.audio_filename and not record.key:
            return False, "missing_audio"

        if context.audio_index:
            candidates = MediaResolver._candidate_names(
                record.audio_filename,
                record.key or None,
            )
            for name in candidates:
                if name in context.audio_index or Path(name).stem in context.audio_index:
                    return True, None

        if MediaResolver.resolve_audio_record(context, record).is_file():
            return True, None
        return False, "missing_audio"

    @staticmethod
    def resolve_audio(context: SplitContext, filename: str, key: str | None = None) -> Path:
        return MediaResolver.resolve_audio(context, filename, key=key)


class AudioDurationValidator(BaseValidator):
    """Reject corrupt or too short audio clips."""

    def __init__(
        self,
        min_duration_sec: float,
        max_duration_sec: float | None = None,
        verify_audio: bool = True,
    ) -> None:
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.verify_audio = verify_audio
        self._audio_validator = AudioFileValidator()

    def validate(self, record: AfrivoiceRecord, context: SplitContext) -> tuple[bool, str | None]:
        duration = self._duration(record, context)
        if duration is None:
            return False, "corrupt_audio"
        if duration < self.min_duration_sec:
            return False, "too_short"
        if self.max_duration_sec is not None and duration > self.max_duration_sec:
            return False, "too_long"
        return True, None

    def _duration(self, record: AfrivoiceRecord, context: SplitContext) -> float | None:
        if not self.verify_audio:
            return record.manifest_duration

        audio_path = self._audio_validator.resolve_audio(
            context,
            record.audio_filename,
            record.key or None,
        )
        return get_audio_duration(audio_path, record)


class AlignmentValidator(BaseValidator):
    """
    Verify transcript-audio alignment using duration and speech-rate heuristics.
    """

    def __init__(self, duration_tolerance_sec: float = ALIGNMENT_DURATION_TOLERANCE_SEC) -> None:
        self.duration_tolerance_sec = duration_tolerance_sec
        self._audio_validator = AudioFileValidator()

    def validate(self, record: AfrivoiceRecord, context: SplitContext) -> tuple[bool, str | None]:
        transcript = record.normalized_transcription or record.transcription
        if not transcript:
            return False, "empty_transcript"

        measured = self._measured_duration(record, context)
        if measured is None:
            return False, "corrupt_audio"

        manifest_duration = record.manifest_duration
        if manifest_duration is not None:
            if abs(measured - manifest_duration) > self.duration_tolerance_sec:
                return False, "misaligned"

        chars_per_sec = len(transcript) / measured
        if chars_per_sec < MIN_CHARS_PER_SEC or chars_per_sec > MAX_CHARS_PER_SEC:
            return False, "misaligned"

        return True, None

    def _measured_duration(self, record: AfrivoiceRecord, context: SplitContext) -> float | None:
        audio_path = self._audio_validator.resolve_audio(
            context,
            record.audio_filename,
            record.key or None,
        )
        return get_audio_duration(audio_path, record)
