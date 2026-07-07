"""
Robust audio duration probing for Afrivoice .webm files.
"""
from __future__ import annotations

from pathlib import Path

from .ffmpeg_setup import configure_ffmpeg, probe_duration
from .models import AfrivoiceRecord


def get_audio_duration(path: Path, record: AfrivoiceRecord | None = None) -> float | None:
    """
    Return clip duration in seconds.

    Tries ffmpeg first (best for .webm), then librosa, then manifest duration.
    """
    configure_ffmpeg()

    duration = probe_duration(path)
    if duration is not None:
        return duration

    try:
        import librosa

        return float(librosa.get_duration(path=path))
    except Exception:
        pass

    if record is not None and record.manifest_duration is not None and path.is_file():
        return record.manifest_duration

    return None
