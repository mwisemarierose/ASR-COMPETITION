"""
Discover dataset split folders for Afrivoice and Anv-ke languages.
"""
from __future__ import annotations

import re
from pathlib import Path

from .archive_extractor import TarXzArchiveExtractor
from .config import (
    ANV_SPLITS,
    ANV_STYLES,
    AUDIO_DIRNAMES,
    DOMAINS,
    MANIFEST_GLOBS,
    SPLITS,
    PipelineConfig,
)
from .models import SplitContext

_SWAHILI_SPLIT_PATTERN = re.compile(r"^(.+)_swahili_(train|dev|test)$")
AFRIVOICE_FLAT_SPLIT = "all"


class DatasetDiscovery:
    """Find split folders under the dataset root for any supported dataset type."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def list_available_splits(self) -> list[tuple[str, str]]:
        """Return available (group, split) pairs — domain/language + split."""
        if self.config.is_anv:
            return self._list_anv_splits()
        return self._list_afrivoice_splits()

    def iter_targets(
        self,
        domain: str | None = None,
        split: str | None = None,
        style: str | None = None,
    ):
        if self.config.is_anv:
            yield from self._iter_anv_targets(split=split, style=style)
            return

        if self.config.is_swahili_domain_layout:
            if domain and split:
                yield self.resolve_afrivoice_split(domain, split)
                return
            for dom, spl in self._list_afrivoice_splits():
                if domain and dom != domain:
                    continue
                if split and spl != split:
                    continue
                yield self.resolve_afrivoice_split(dom, spl)
            return

        language = self.config.afrivoice_language
        if domain and domain != language:
            return
        if split and split != AFRIVOICE_FLAT_SPLIT:
            return
        yield self.resolve_afrivoice_language_folder(language)

    def resolve_afrivoice_split(self, domain: str, split: str) -> SplitContext:
        folder = self.config.split_folder_path(domain, split)
        manifests = self._find_manifests(folder)
        audio_dir = self._find_audio_dir(folder)
        return SplitContext(
            domain=domain,
            split=split,
            language=self.config.afrivoice_language,
            folder=folder,
            audio_dir=audio_dir,
            manifest_paths=manifests,
            audio_archives=TarXzArchiveExtractor.find_archives(audio_dir),
        )

    def resolve_afrivoice_language_folder(self, language: str) -> SplitContext:
        folder = self.config.afrivoice_language_folder()
        manifests = self._find_manifests(folder)
        audio_dir = self._find_audio_dir(folder)
        return SplitContext(
            domain=language,
            split=AFRIVOICE_FLAT_SPLIT,
            language=language,
            folder=folder,
            audio_dir=audio_dir,
            manifest_paths=manifests,
            audio_archives=TarXzArchiveExtractor.find_archives(audio_dir),
        )

    def _list_afrivoice_splits(self) -> list[tuple[str, str]]:
        if not self.config.is_swahili_domain_layout:
            return [(self.config.afrivoice_language, AFRIVOICE_FLAT_SPLIT)]

        found: list[tuple[str, str]] = []
        root = self.config.dataset_root
        if not root.is_dir():
            return found

        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            match = _SWAHILI_SPLIT_PATTERN.match(path.name)
            if not match:
                continue
            found.append((match.group(1), match.group(2)))
        return found

    @staticmethod
    def _find_manifests(folder: Path) -> list[Path]:
        if not folder.is_dir():
            return []
        manifests: list[Path] = []
        for pattern in MANIFEST_GLOBS:
            manifests.extend(sorted(folder.glob(pattern)))
        return manifests

    @staticmethod
    def _find_audio_dir(folder: Path) -> Path | None:
        if not folder.is_dir():
            return None
        for dirname in AUDIO_DIRNAMES:
            audio_dir = folder / dirname
            if audio_dir.is_dir():
                return audio_dir
        return folder / AUDIO_DIRNAMES[0]

    def _list_anv_splits(self) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        root = self.config.language_dataset_root()
        if not root.is_dir() or not self.config.language:
            return found

        for split_dir in sorted(root.iterdir()):
            if split_dir.is_dir() and split_dir.name in ANV_SPLITS:
                found.append((self.config.language, split_dir.name))
        return found

    def _iter_anv_targets(
        self,
        split: str | None = None,
        style: str | None = None,
    ):
        root = self.config.language_dataset_root()
        if not root.is_dir() or not self.config.language:
            return

        for split_dir in sorted(root.iterdir()):
            if not split_dir.is_dir():
                continue
            spl = split_dir.name
            if spl not in ANV_SPLITS:
                continue
            if split and spl != split:
                continue

            for style_dir in sorted(split_dir.iterdir()):
                if not style_dir.is_dir():
                    continue
                sty = style_dir.name
                if sty not in ANV_STYLES:
                    continue
                if style and sty != style:
                    continue

                parquet_paths = self._parquet_shards(style_dir)
                if not parquet_paths:
                    continue

                files_dir = style_dir / "files"
                meta_csv = files_dir / "meta.csv" if (files_dir / "meta.csv").is_file() else None
                transcripts_csv = (
                    files_dir / "transcripts.csv"
                    if (files_dir / "transcripts.csv").is_file()
                    else None
                )
                yield SplitContext(
                    language=self.config.language,
                    split=spl,
                    style=sty,
                    folder=style_dir,
                    parquet_paths=tuple(parquet_paths),
                    meta_csv=meta_csv,
                    transcripts_csv=transcripts_csv,
                )

    @staticmethod
    def _parquet_shards(style_dir: Path) -> list[Path]:
        audio_dir = style_dir / "audios"
        if not audio_dir.is_dir():
            return []
        return sorted(path for path in audio_dir.glob("*.parquet") if path.is_file())

    def planned_splits(
        self,
        domain: str | None = None,
        split: str | None = None,
    ) -> list[tuple[str, str]]:
        if not self.config.is_swahili_domain_layout:
            language = self.config.afrivoice_language
            if domain and domain != language:
                return []
            if split and split != AFRIVOICE_FLAT_SPLIT:
                return []
            return [(language, AFRIVOICE_FLAT_SPLIT)]

        if domain and split:
            return [(domain, split)]
        if domain:
            return [(domain, s) for s in SPLITS]
        if split:
            return [(d, split) for d in DOMAINS]
        return self.list_available_splits()
