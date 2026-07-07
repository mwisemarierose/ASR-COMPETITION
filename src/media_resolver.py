"""
Resolve audio paths from loose files or extracted tar.xz archives.

Afrivoice audio is stored as hash-named .webm files after extraction, e.g.:
    MsYzAS3P092NBPdEBTMa.webm

Manifest rows reference them via audio_filepath and/or key.
"""
from __future__ import annotations

from pathlib import Path

from .config import AUDIO_EXTENSIONS, DEFAULT_AUDIO_EXTENSION
from .models import AfrivoiceRecord, SplitContext


class MediaResolver:
    """Locate audio files across raw folders and extracted archives."""

    @staticmethod
    def resolve_audio(
        context: SplitContext,
        filename: str,
        key: str | None = None,
    ) -> Path:
        if not filename and not key:
            return context.audio_dir / "missing"

        search_dirs = MediaResolver._search_dirs(context)
        candidates = MediaResolver._candidate_names(filename, key)

        for directory in search_dirs:
            for name in candidates:
                candidate = directory / name
                if candidate.is_file():
                    return candidate

        for directory in search_dirs:
            for name in candidates:
                matches = sorted(directory.rglob(Path(name).name))
                if matches:
                    return matches[0]

        fallback_name = candidates[0] if candidates else filename
        return search_dirs[0] / fallback_name if search_dirs else Path(fallback_name)

    @staticmethod
    def resolve_audio_record(context: SplitContext, record: AfrivoiceRecord) -> Path:
        return MediaResolver.resolve_audio(
            context=context,
            filename=record.audio_filename,
            key=record.key or None,
        )

    @staticmethod
    def _candidate_names(filename: str, key: str | None) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name and name not in seen:
                seen.add(name)
                names.append(name)

        if filename:
            add(filename)
            add(Path(filename).name)

        if key:
            add(f"{key}{DEFAULT_AUDIO_EXTENSION}")
            for ext in AUDIO_EXTENSIONS:
                add(f"{key}{ext}")

        return names

    @staticmethod
    def _search_dirs(context: SplitContext) -> tuple[Path, ...]:
        dirs: list[Path] = []
        if context.extracted_audio_dir and context.extracted_audio_dir.is_dir():
            dirs.append(context.extracted_audio_dir)
        if context.audio_dir.is_dir():
            dirs.append(context.audio_dir)
        dirs.append(context.folder)
        return tuple(dirs)
