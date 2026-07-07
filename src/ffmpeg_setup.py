"""
Make ffmpeg available without sudo by using a user-local pip bundle.

Install with:
    pip install imageio-ffmpeg

This module prepends that binary to PATH so librosa/audioread can read .webm files.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def configure_ffmpeg() -> str | None:
    """
    Ensure ffmpeg is on PATH.

    Order:
    1. Existing system ffmpeg
    2. imageio-ffmpeg pip bundle (no sudo required)
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not bundled.is_file():
        return None

    bundled_dir = str(bundled.parent)
    current_path = os.environ.get("PATH", "")
    if bundled_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = bundled_dir + os.pathsep + current_path

    return str(bundled)


def ffmpeg_status() -> str:
    """Human-readable ffmpeg availability message."""
    path = configure_ffmpeg()
    if not path:
        return (
            "ffmpeg not found. Install one of:\n"
            "  pip install imageio-ffmpeg   # no sudo\n"
            "  conda install -c conda-forge ffmpeg"
        )
    return f"ffmpeg available: {path}"
