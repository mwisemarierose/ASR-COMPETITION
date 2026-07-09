"""
Transcript normalization for Swahili ASR training.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .models import AfrivoiceRecord

_CS_TAG_PATTERN = re.compile(r"\[cs\](.*?)\[/cs\]", re.IGNORECASE)
_CS_TOKEN_PATTERN = re.compile(r"cs(\w+)cs", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


class BaseTranscriptCleaner(ABC):
    """Normalize transcript text for downstream ASR models."""

    @abstractmethod
    def clean(self, record: AfrivoiceRecord) -> str:
        raise NotImplementedError


class SwahiliTranscriptCleaner(BaseTranscriptCleaner):
    """
    Prefer normalized_transcription, then fall back to transcription.
    Strips code-switch markers and collapses whitespace.
    """

    def clean(self, record: AfrivoiceRecord) -> str:
        text = record.normalized_transcription or record.transcription
        text = _CS_TAG_PATTERN.sub(r"\1", text)
        text = _CS_TOKEN_PATTERN.sub(r"\1", text)
        text = text.lower().strip()
        text = _WHITESPACE_PATTERN.sub(" ", text)
        return text


_DISFLUENCY_PATTERN = re.compile(
    r"\[(pause|sigh|laugh|breath|noise|silence)\]",
    re.IGNORECASE,
)
_PUNCTUATION_PATTERN = re.compile(r'[.,;:?!"/\\]')


def clean_anv_transcript(text: str) -> str:
    """Normalize Anv-ke transcripts (lowercase, strip disfluency markers)."""
    if not text:
        return ""

    cleaned = _CS_TAG_PATTERN.sub(r"\1", text)
    cleaned = _DISFLUENCY_PATTERN.sub(" ", cleaned)
    cleaned = _PUNCTUATION_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.lower().strip()
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned
