"""
Resolve audio/image paths from loose files or extracted tar.xz archives.

Afrivoice audio is stored as hash-named .webm files after extraction, e.g.:
    MsYzAS3P092NBPdEBTMa.webm

Manifest rows reference them via audio_filepath and/or key.
"""
from __future__ import annotations

from pathlib import Path

from .config import AUDIO_EXTENSIONS, DEFAULT_AUDIO_EXTENSION
from .models import AfrivoiceRecord, SplitContext


class MediaResolver:
    """Locate media files across raw folders and extracted archives."""

    @staticmethod
    def resolve_audio(
        context: SplitContext,
        filename: str,
        key: str | None = None,
    ) -> Path:
        return MediaResolver._resolve(
            context=context,
            filename=filename,
            media_type="audio",
            key=key,
        )

    @staticmethod
    def resolve_audio_record(context: SplitContext, record: AfrivoiceRecord) -> Path:
        return MediaResolver.resolve_audio(
            context=context,
            filename=record.audio_filename,
            key=record.key or None,
        )

    @staticmethod
    def resolve_image(context: SplitContext, filename: str) -> Path:
        return MediaResolver._resolve(context, filename, media_type="image")

    @staticmethod
    def _resolve(
        context: SplitContext,
        filename: str,
        media_type: str,
        key: str | None = None,
    ) -> Path:
        if not filename and not key:
            fallback = context.audio_dir if media_type == "audio" else context.image_dir
            return fallback / "missing"

        search_dirs = MediaResolver._search_dirs(context, media_type)
        candidates = MediaResolver._candidate_names(filename, key, media_type)

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
    def _candidate_names(
        filename: str,
        key: str | None,
        media_type: str,
    ) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name and name not in seen:
                seen.add(name)
                names.append(name)

        if filename:
            add(filename)
            add(Path(filename).name)

        if media_type == "audio" and key:
            add(f"{key}{DEFAULT_AUDIO_EXTENSION}")
            for ext in AUDIO_EXTENSIONS:
                add(f"{key}{ext}")

        return names

    @staticmethod
    def _search_dirs(context: SplitContext, media_type: str) -> tuple[Path, ...]:
        if media_type == "audio":
            extracted = context.extracted_audio_dir
            raw = context.audio_dir
        else:
            extracted = context.extracted_image_dir
            raw = context.image_dir

        dirs: list[Path] = []
        if extracted and extracted.is_dir():
            dirs.append(extracted)
        if raw.is_dir():
            dirs.append(raw)
        dirs.append(context.folder)
        return tuple(dirs)
