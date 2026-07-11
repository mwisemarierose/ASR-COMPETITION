"""
Audio decoding for file-based (Afrivoice) and Parquet-embedded (Anv-ke) clips.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import av
import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .config import SAMPLE_RATE
from .ffmpeg_setup import configure_ffmpeg, probe_duration
from .models import AfrivoiceRecord

PARQUET_AUDIO_COLUMNS = ("audio", "bytes", "audio_bytes")
PARQUET_TEXT_COLUMNS = ("transcription", "transcript", "text", "sentence")
PARQUET_ID_COLUMNS = ("recorder_uuid", "mediaPathId", "media_path_id", "id", "key")

# Per-worker cache: parquet path -> audio column name (schema probe once per file)
_PARQUET_AUDIO_COL_CACHE: dict[str, str] = {}


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


def audio_bytes_from_struct(audio_value: Any) -> bytes | None:
    """Extract raw bytes from HF-style audio struct {"bytes": ..., "path": ...}."""
    if audio_value is None:
        return None
    if isinstance(audio_value, (bytes, bytearray)):
        return bytes(audio_value)
    if isinstance(audio_value, dict):
        payload = audio_value.get("bytes")
        if payload:
            return bytes(payload)
    return None


def _decode_audio_bytes(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode embedded audio bytes; prefer soundfile/librosa before PyAV."""
    if audio_bytes[:4] in (b"RIFF", b"fLaC", b"OggS") or audio_bytes[:3] == b"ID3":
        try:
            audio, file_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            if file_sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sample_rate)
            return audio
        except Exception:
            pass

    try:
        audio, file_sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
        return np.asarray(audio, dtype=np.float32)
    except Exception:
        pass

    if not audio_bytes:
        raise RuntimeError("Empty audio bytes")

    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []

    try:
        with av.open(io.BytesIO(audio_bytes), metadata_errors="ignore") as container:
            if not container.streams.audio:
                raise RuntimeError("No audio stream in parquet bytes")

            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(_frame_to_mono_float32(resampled))

            for resampled in resampler.resample(None):
                chunks.append(_frame_to_mono_float32(resampled))
    except av.error.FFmpegError as exc:
        raise RuntimeError(f"Cannot decode parquet audio bytes: {exc}") from exc

    if not chunks:
        raise RuntimeError("No audio frames decoded from parquet bytes")

    return np.concatenate(chunks).astype(np.float32, copy=False)


def duration_from_audio_bytes(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    audio = _decode_audio_bytes(audio_bytes, sample_rate=sample_rate)
    return float(len(audio) / sample_rate)


def load_parquet_row_audio(
    parquet_path: Path,
    row_index: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Load one clip without reading an entire Parquet row group into memory."""
    audio_bytes = load_parquet_row_audio_bytes(parquet_path, row_index)
    return _decode_audio_bytes(audio_bytes, sample_rate=sample_rate)


def load_parquet_row_audio_bytes(parquet_path: Path, row_index: int) -> bytes:
    audio_col = _resolve_parquet_audio_column(parquet_path)
    pf = pq.ParquetFile(parquet_path)

    cumulative = 0
    for rg in range(pf.num_row_groups):
        n_rows = pf.metadata.row_group(rg).num_rows
        if row_index < cumulative + n_rows:
            local_idx = row_index - cumulative
            for batch in pf.iter_batches(
                batch_size=1,
                row_groups=[rg],
                columns=[audio_col],
            ):
                if local_idx == 0:
                    audio_value = batch.column(0)[0].as_py()
                    audio_bytes = audio_bytes_from_struct(audio_value)
                    if not audio_bytes:
                        raise RuntimeError(f"Missing audio bytes at {parquet_path}:{row_index}")
                    return audio_bytes
                local_idx -= 1
            break
        cumulative += n_rows

    raise RuntimeError(f"Row {row_index} out of range for {parquet_path}")


def load_parquet_audio_column(parquet_path: Path):
    """Load the full audio column from a shard (use when processing many rows)."""
    audio_col = _resolve_parquet_audio_column(parquet_path)
    return pq.read_table(parquet_path, columns=[audio_col]).column(0)


def _resolve_parquet_audio_column(parquet_path: Path) -> str:
    key = str(parquet_path.resolve())
    cached = _PARQUET_AUDIO_COL_CACHE.get(key)
    if cached:
        return cached

    schema_names = pq.read_schema(parquet_path).names
    for candidate in PARQUET_AUDIO_COLUMNS:
        if candidate in schema_names:
            _PARQUET_AUDIO_COL_CACHE[key] = candidate
            return candidate
    raise RuntimeError(f"No audio column found in {parquet_path}; columns={schema_names}")


def resolve_parquet_text_column(schema_names: list[str]) -> str | None:
    for candidate in PARQUET_TEXT_COLUMNS:
        if candidate in schema_names:
            return candidate
    return None


def resolve_parquet_id_column(schema_names: list[str]) -> str | None:
    for candidate in PARQUET_ID_COLUMNS:
        if candidate in schema_names:
            return candidate
    return None
