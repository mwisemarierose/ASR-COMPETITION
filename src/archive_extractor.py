"""
Extract audio_*.tar.xz and image_*.tar.xz archives from Afrivoice splits.
"""
from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig
from .models import SplitContext

ARCHIVE_GLOB = "*.tar.xz"


@dataclass
class ExtractionReport:
    """Summary of archive extraction for one media type."""

    media_type: str
    archives: list[str] = field(default_factory=list)
    target_dir: str = ""
    extracted_files: int = 0
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "media_type": self.media_type,
            "archives": self.archives,
            "target_dir": self.target_dir,
            "extracted_files": self.extracted_files,
            "skipped": self.skipped,
        }


class TarXzArchiveExtractor:
    """Extract .tar.xz media bundles into a reusable cache directory."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def prepare_split(self, context: SplitContext) -> list[ExtractionReport]:
        """Extract audio/image archives for one split if needed."""
        reports: list[ExtractionReport] = []

        audio_report = self._prepare_media(
            context=context,
            media_type="audio",
            source_dir=context.audio_dir,
            archives=context.audio_archives,
        )
        if audio_report:
            reports.append(audio_report)
            if audio_report.target_dir:
                context.extracted_audio_dir = Path(audio_report.target_dir)

        image_report = self._prepare_media(
            context=context,
            media_type="image",
            source_dir=context.image_dir,
            archives=context.image_archives,
        )
        if image_report:
            reports.append(image_report)
            if image_report.target_dir:
                context.extracted_image_dir = Path(image_report.target_dir)

        return reports

    def _prepare_media(
        self,
        context: SplitContext,
        media_type: str,
        source_dir: Path,
        archives: list[Path],
    ) -> ExtractionReport | None:
        if not archives:
            return None

        target_dir = self._cache_dir(context, media_type)
        marker = target_dir / ".extracted.json"

        if self.config.skip_extract:
            if not marker.is_file():
                raise FileNotFoundError(
                    f"Extracted {media_type} cache missing for {context.domain}/{context.split}: "
                    f"{target_dir}. Run without --skip-extract first."
                )
            context_data = json.loads(marker.read_text(encoding="utf-8"))
            return ExtractionReport(
                media_type=media_type,
                archives=context_data.get("archives", []),
                target_dir=str(target_dir),
                extracted_files=context_data.get("extracted_files", 0),
                skipped=True,
            )

        if not self.config.force_extract and marker.is_file() and self._cache_is_current(marker, archives):
            context_data = json.loads(marker.read_text(encoding="utf-8"))
            return ExtractionReport(
                media_type=media_type,
                archives=[path.name for path in archives],
                target_dir=str(target_dir),
                extracted_files=context_data.get("extracted_files", 0),
                skipped=True,
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = 0

        for archive in archives:
            with tarfile.open(archive, mode="r:xz") as handle:
                handle.extractall(path=target_dir)
                extracted_files += sum(1 for member in handle.getmembers() if member.isfile())

        marker.write_text(
            json.dumps(
                {
                    "archives": [
                        {"name": path.name, "size": path.stat().st_size, "mtime": path.stat().st_mtime}
                        for path in archives
                    ],
                    "extracted_files": extracted_files,
                    "source_dir": str(source_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return ExtractionReport(
            media_type=media_type,
            archives=[path.name for path in archives],
            target_dir=str(target_dir),
            extracted_files=extracted_files,
            skipped=False,
        )

    def _cache_dir(self, context: SplitContext, media_type: str) -> Path:
        return (
            self.config.extract_cache_root
            / context.domain
            / context.split
            / media_type
        )

    @staticmethod
    def _cache_is_current(marker: Path, archives: list[Path]) -> bool:
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))["archives"]
        except (json.JSONDecodeError, KeyError, OSError):
            return False

        if len(saved) != len(archives):
            return False

        for entry, archive in zip(saved, archives):
            if entry.get("name") != archive.name:
                return False
            stat = archive.stat()
            if entry.get("size") != stat.st_size or entry.get("mtime") != stat.st_mtime:
                return False
        return True

    @staticmethod
    def find_archives(media_dir: Path) -> list[Path]:
        if not media_dir.is_dir():
            return []
        return sorted(media_dir.glob(ARCHIVE_GLOB))
