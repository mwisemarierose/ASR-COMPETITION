"""
Shared log-mel feature extraction for Afrivoice WAV and in-memory audio.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


def audio_to_log_mel(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Convert mono float32 audio to an 80-bin log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return librosa.power_to_db(mel, ref=np.max)


def wav_path_to_log_mel(wav_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    return audio_to_log_mel(audio, sample_rate=sr)
