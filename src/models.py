"""
Data models for the Afrivoice cleaning pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AfrivoiceRecord:
    """One JSONL row from an Afrivoice_Swahili manifest."""

    raw: dict[str, Any]
    source_manifest: Path
    line_number: int

    @property
    def key(self) -> str:
        return str(self.raw.get("key", ""))

    @property
    def transcription(self) -> str:
        return str(self.raw.get("transcription", "") or "").strip()

    @property
    def normalized_transcription(self) -> str:
        return str(self.raw.get("normalized_transcription", "") or "").strip()

    @property
    def audio_filename(self) -> str:
        return str(self.raw.get("audio_filepath", "") or "").strip()

    @property
    def image_filename(self) -> str:
        return str(self.raw.get("image_filepath", "") or "").strip()

    @property
    def manifest_duration(self) -> float | None:
        value = self.raw.get("duration")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


@dataclass
class SplitContext:
    """Resolved paths for one domain/split folder."""

    domain: str
    split: str
    folder: Path
    audio_dir: Path
    image_dir: Path
    manifest_paths: list[Path] = field(default_factory=list)
    audio_archives: list[Path] = field(default_factory=list)
    image_archives: list[Path] = field(default_factory=list)
    extracted_audio_dir: Path | None = None
    extracted_image_dir: Path | None = None


@dataclass
class FilterStats:
    """Counters for one split cleaning run."""

    input_rows: int = 0
    empty_transcript: int = 0
    missing_audio: int = 0
    missing_image: int = 0
    corrupt_audio: int = 0
    too_short: int = 0
    too_long: int = 0
    duplicate_key: int = 0
    misaligned: int = 0
    kept: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "empty_transcript": self.empty_transcript,
            "missing_audio": self.missing_audio,
            "missing_image": self.missing_image,
            "corrupt_audio": self.corrupt_audio,
            "too_short": self.too_short,
            "too_long": self.too_long,
            "duplicate_key": self.duplicate_key,
            "misaligned": self.misaligned,
            "kept": self.kept,
        }


@dataclass
class VerifyReport:
    """Read-only integrity report for one split."""

    domain: str
    split: str
    folder: Path
    manifests: list[str] = field(default_factory=list)
    rows: int = 0
    missing_audio: int = 0
    empty_transcripts: int = 0
    audio_archives: list[str] = field(default_factory=list)
    image_archives: list[str] = field(default_factory=list)
    extraction: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # Structural problems block the run; bad rows are filtered during cleaning.
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "split": self.split,
            "folder": str(self.folder),
            "manifests": self.manifests,
            "rows": self.rows,
            "missing_audio": self.missing_audio,
            "empty_transcripts": self.empty_transcripts,
            "audio_archives": self.audio_archives,
            "image_archives": self.image_archives,
            "extraction": self.extraction,
            "issues": self.issues,
            "ok": self.ok,
        }
