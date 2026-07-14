"""
Extract audio_*.tar.xz archives from Afrivoice splits.
"""
from __future__ import annotations

import json
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .config import PipelineConfig
from .models import SplitContext

ARCHIVE_GLOB = "*.tar.xz"


@dataclass
class ExtractionReport:
    """Summary of archive extraction for one split."""

    archives: list[str] = field(default_factory=list)
    target_dir: str = ""
    extracted_files: int = 0
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "archives": self.archives,
            "target_dir": self.target_dir,
            "extracted_files": self.extracted_files,
            "skipped": self.skipped,
        }


class TarXzArchiveExtractor:
    """Extract audio .tar.xz bundles into a reusable cache directory."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def prepare_split(self, context: SplitContext) -> ExtractionReport | None:
        """Extract audio archives for one split if needed."""
        return self._prepare_audio(context)

    def _prepare_audio(self, context: SplitContext) -> ExtractionReport | None:
        archives = context.audio_archives
        if not archives:
            return None

        target_dir = self._cache_dir(context)
        marker = target_dir / ".extracted.json"

        if self.config.skip_extract:
            if not marker.is_file():
                raise FileNotFoundError(
                    f"Extracted audio cache missing for {context.domain}/{context.split}: "
                    f"{target_dir}. Run without --skip-extract first."
                )
            context_data = json.loads(marker.read_text(encoding="utf-8"))
            report = ExtractionReport(
                archives=context_data.get("archives", []),
                target_dir=str(target_dir),
                extracted_files=context_data.get("extracted_files", 0),
                skipped=True,
            )
            context.extracted_audio_dir = target_dir
            return report

        if not self.config.force_extract and marker.is_file() and self._cache_is_current(marker, archives):
            context_data = json.loads(marker.read_text(encoding="utf-8"))
            report = ExtractionReport(
                archives=[path.name for path in archives],
                target_dir=str(target_dir),
                extracted_files=context_data.get("extracted_files", 0),
                skipped=True,
            )
            context.extracted_audio_dir = target_dir
            return report

        target_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = 0
        total_archives = len(archives)

        for index, archive in enumerate(
            tqdm(archives, desc="  extract archives", unit="archive"),
            start=1,
        ):
            archive_files = 0
            started = time.monotonic()
            tqdm.write(f"    [{index}/{total_archives}] {archive.name} ...")
            with tarfile.open(archive, mode="r:xz") as handle:
                archive_files = sum(1 for member in handle.getmembers() if member.isfile())
                handle.extractall(path=target_dir)
            extracted_files += archive_files
            elapsed = time.monotonic() - started
            tqdm.write(
                f"    done {archive.name}: {archive_files} files in {elapsed:.0f}s "
                f"(running total {extracted_files})"
            )

        marker.write_text(
            json.dumps(
                {
                    "archives": [
                        {"name": path.name, "size": path.stat().st_size, "mtime": path.stat().st_mtime}
                        for path in archives
                    ],
                    "extracted_files": extracted_files,
                    "source_dir": str(context.audio_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        context.extracted_audio_dir = target_dir
        return ExtractionReport(
            archives=[path.name for path in archives],
            target_dir=str(target_dir),
            extracted_files=extracted_files,
            skipped=False,
        )

    def _cache_dir(self, context: SplitContext) -> Path:
        return (
            self.config.extract_cache_root
            / context.domain
            / context.split
            / "audio"
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
    def find_archives(audio_dir: Path) -> list[Path]:
        if not audio_dir.is_dir():
            return []
        return sorted(audio_dir.glob(ARCHIVE_GLOB))
