"""
Discover Afrivoice_Swahili domain/split folders on disk.
"""
from __future__ import annotations

import re
from pathlib import Path

from .archive_extractor import TarXzArchiveExtractor
from .config import AUDIO_DIRNAME, DOMAINS, IMAGE_DIRNAME, MANIFEST_GLOB, SPLITS, PipelineConfig
from .models import SplitContext

_SPLIT_PATTERN = re.compile(r"^(.+)_swahili_(train|dev|test)$")


class DatasetDiscovery:
    """Find split folders under the dataset root."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def list_available_splits(self) -> list[tuple[str, str]]:
        """Return (domain, split) pairs that exist on disk."""
        found: list[tuple[str, str]] = []
        root = self.config.dataset_root
        if not root.is_dir():
            return found

        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            match = _SPLIT_PATTERN.match(path.name)
            if not match:
                continue
            domain, split = match.group(1), match.group(2)
            found.append((domain, split))
        return found

    def iter_targets(
        self,
        domain: str | None = None,
        split: str | None = None,
    ):
        """
        Yield SplitContext objects for requested domain/split filters.
        If no filter is given, yields every folder found on disk.
        """
        if domain and split:
            yield self.resolve_split(domain, split)
            return

        available = self.list_available_splits()
        for dom, spl in available:
            if domain and dom != domain:
                continue
            if split and spl != split:
                continue
            yield self.resolve_split(dom, spl)

    def resolve_split(self, domain: str, split: str) -> SplitContext:
        folder = self.config.split_folder_path(domain, split)
        manifests = sorted(folder.glob(MANIFEST_GLOB)) if folder.is_dir() else []
        audio_dir = folder / AUDIO_DIRNAME
        image_dir = folder / IMAGE_DIRNAME
        return SplitContext(
            domain=domain,
            split=split,
            folder=folder,
            audio_dir=audio_dir,
            image_dir=image_dir,
            manifest_paths=manifests,
            audio_archives=TarXzArchiveExtractor.find_archives(audio_dir),
            image_archives=TarXzArchiveExtractor.find_archives(image_dir),
        )

    def planned_splits(
        self,
        domain: str | None = None,
        split: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return explicit domain/split pairs when filters are set."""
        if domain and split:
            return [(domain, split)]
        if domain:
            return [(domain, s) for s in SPLITS]
        if split:
            return [(d, split) for d in DOMAINS]
        return self.list_available_splits()
