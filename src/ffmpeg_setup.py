"""
Make ffmpeg available without sudo by using a user-local pip bundle.

imageio-ffmpeg installs a binary named e.g. ffmpeg-linux-x86_64-v7.1,
but audioread/librosa look for a command literally called `ffmpeg`.
This module creates a symlink and prepends it to PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import PROJECT_ROOT

_FFMPEG_BIN_DIR = PROJECT_ROOT / ".bin"


def configure_ffmpeg() -> str | None:
    """
    Ensure a command named `ffmpeg` is on PATH.

    Order:
    1. Existing system ffmpeg
    2. Symlink to imageio-ffmpeg bundle (no sudo required)
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        _prepend_path(str(Path(system_ffmpeg).parent))
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not bundled.is_file():
        return None

    _FFMPEG_BIN_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_link = _FFMPEG_BIN_DIR / "ffmpeg"

    if ffmpeg_link.is_symlink() or ffmpeg_link.exists():
        try:
            if ffmpeg_link.resolve() != bundled.resolve():
                ffmpeg_link.unlink()
        except FileNotFoundError:
            ffmpeg_link.unlink(missing_ok=True)

    if not ffmpeg_link.exists():
        ffmpeg_link.symlink_to(bundled)

    _prepend_path(str(_FFMPEG_BIN_DIR))
    return str(ffmpeg_link.resolve())


def _prepend_path(directory: str) -> None:
    current_path = os.environ.get("PATH", "")
    if directory not in current_path.split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + current_path


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


def probe_duration(path: Path) -> float | None:
    """Read audio duration using ffmpeg (works for .webm)."""
    ffmpeg = configure_ffmpeg()
    if not ffmpeg or not path.is_file():
        return None

    proc = subprocess.run(
        [ffmpeg, "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stderr + proc.stdout

    for line in output.splitlines():
        if "Duration:" not in line:
            continue
        try:
            duration_token = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
            hours, minutes, seconds = duration_token.split(":")
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except (IndexError, ValueError):
            continue

    return None
