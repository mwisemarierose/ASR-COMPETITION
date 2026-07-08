"""
Robust audio duration probing and decoding for Afrivoice .webm files.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE
from .ffmpeg_setup import configure_ffmpeg, probe_duration
from .models import AfrivoiceRecord


def _frame_to_mono_float32(frame: av.AudioFrame) -> np.ndarray:
    audio = frame.to_ndarray()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return np.asarray(audio, dtype=np.float32)


def _load_audio_with_av(path: Path, sample_rate: int) -> np.ndarray:
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []

    with av.open(str(path), metadata_errors="ignore") as container:
        if not container.streams.audio:
            raise RuntimeError(f"No audio stream found in {path}")

        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(_frame_to_mono_float32(resampled))

        for resampled in resampler.resample(None):
            chunks.append(_frame_to_mono_float32(resampled))

    if not chunks:
        raise RuntimeError(f"Cannot read audio {path}: no audio frames decoded")

    return np.concatenate(chunks).astype(np.float32, copy=False)


def _probe_duration_av(path: Path) -> float | None:
    try:
        with av.open(str(path), metadata_errors="ignore") as container:
            if container.duration:
                return float(container.duration * av.time_base)

            if container.streams.audio:
                stream = container.streams.audio[0]
                if stream.duration:
                    return float(stream.duration * stream.time_base)
    except Exception:
        return None

    return None


def load_audio_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode audio to mono float32 at the target sample rate."""
    suffix = path.suffix.lower()

    if suffix == ".wav":
        audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sample_rate:
            return _load_audio_with_av(path, sample_rate)
        return np.asarray(audio, dtype=np.float32)

    return _load_audio_with_av(path, sample_rate)


def get_audio_duration(
    path: Path, record: AfrivoiceRecord | None = None
) -> float | None:
    """
    Return clip duration in seconds.

    Tries PyAV first, then ffmpeg, then manifest duration.
    """
    configure_ffmpeg()

    duration = _probe_duration_av(path)
    if duration is not None:
        return duration

    duration = probe_duration(path)
    if duration is not None:
        return duration

    if record is not None and record.manifest_duration is not None and path.is_file():
        return record.manifest_duration

    return None
