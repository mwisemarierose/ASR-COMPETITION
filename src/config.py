"""
Project settings and folder paths for all ASR dataset pipelines.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Afrivoice Swahili (DigitalUmuganda/Afrivoice_Swahili) ---
DOMAINS = ("agriculture", "education", "financial", "government", "health")
SPLITS = ("train", "dev", "test")

# --- Afrivoice languages ---
# Swahili uses domain/split folders; other Afrivoice langs use a flat language folder.
AFRIVOICE_LANGUAGES: dict[str, dict[str, object]] = {
    "swahili": {
        "layout": "domain_split",
        "folder_names": (),
        "display": "Swahili",
    },
    "somali": {
        "layout": "language_flat",
        "folder_names": ("Somali", "somali"),
        "display": "Somali",
    },
    "shona": {
        "layout": "language_flat",
        "folder_names": ("Shona", "shona"),
        "display": "Shona",
    },
    "lingala": {
        "layout": "language_flat",
        "folder_names": ("Lingala", "lingala"),
        "display": "Lingala",
    },
    "fulani": {
        "layout": "language_flat",
        "folder_names": ("Fulani", "fulani", "Fulah"),
        "display": "Fulani",
    },
    "wolof": {
        "layout": "language_flat",
        "folder_names": ("Wolof", "wolof"),
        "display": "Wolof",
    },
    "malagasy": {
        "layout": "language_flat",
        "folder_names": ("Malagasy", "malagasy"),
        "display": "Malagasy",
    },
}

DEFAULT_DATASET_ROOT = Path(
    os.environ.get("DATASET_ROOT", PROJECT_ROOT / "data" / "raw" / "Afrivoice_Swahili")
)

DEFAULT_AFRIVOICE_MULTILANG_ROOT = Path(
    os.environ.get(
        "AFRIVOICE_DATASET_ROOT",
        PROJECT_ROOT / "data" / "raw" / "Afrivoice",
    )
)

# --- Anv-ke languages (Parquet + CSV) ---
ANV_LANGUAGES: dict[str, str] = {
    "kalenjin": "Kalenjin",
    "dholuo": "Dholuo",
    "luo": "Dholuo",
    "kikuyu": "Kikuyu",
    "somali": "Somali",
    "maasai": "Maasai",
}
ANV_SPLITS = ("train", "dev", "test", "dev_test")
ANV_STYLES = ("scripted", "unscripted")

DEFAULT_ANV_DATASET_ROOT = Path(
    os.environ.get("ANV_DATASET_ROOT", PROJECT_ROOT / "data" / "raw" / "Anv-ke")
)

# --- Shared output paths ---
CLEANED_ROOT = PROJECT_ROOT / "data" / "cleaned"
EXTRACTED_ROOT = PROJECT_ROOT / "data" / "extracted"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
STATS_DIR = PROJECT_ROOT / "outputs" / "statistics"
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"

# Cleaning thresholds (requirement: remove audio < 2 seconds)
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = None
ALIGNMENT_DURATION_TOLERANCE_SEC = 2.0
MIN_CHARS_PER_SEC = 2.0
MAX_CHARS_PER_SEC = 30.0

# Preprocessing + features (requirement: 16 kHz, 80 mel bins)
SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160

# Anv-ke extract: cap parallel batch size and full-shard reads to avoid OOM on HPC.
ANV_CLIPS_PER_BATCH = 50
MAX_PARQUET_COLUMN_LOAD_BYTES = 150 * 1024 * 1024

# Afrivoice manifest + audio layout
AUDIO_DIRNAMES = ("audio", "audio_shards")
MANIFEST_GLOBS = ("manifest_*.jsonl", "manifest_*.json")
# Backward-compatible aliases
AUDIO_DIRNAME = AUDIO_DIRNAMES[0]
MANIFEST_GLOB = MANIFEST_GLOBS[0]
AUDIO_EXTENSIONS = (".webm", ".wav", ".mp3", ".ogg", ".m4a")
DEFAULT_AUDIO_EXTENSION = ".webm"


@dataclass
class PipelineConfig:
    """Runtime configuration for any supported dataset pipeline."""

    dataset_type: str = "afrivoice"
    dataset_root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT)
    language: str | None = None
    output_root: Path = field(default_factory=lambda: CLEANED_ROOT)
    extract_cache_root: Path = field(default_factory=lambda: EXTRACTED_ROOT)
    processed_root: Path = field(default_factory=lambda: PROCESSED_ROOT)
    features_dir: Path = field(default_factory=lambda: FEATURES_DIR)
    stats_dir: Path = field(default_factory=lambda: STATS_DIR)
    min_duration_sec: float = MIN_DURATION_SEC
    max_duration_sec: float | None = MAX_DURATION_SEC
    verify_audio: bool = True
    dry_run: bool = False
    skip_extract: bool = False
    force_extract: bool = False
    verify_alignment: bool = True
    skip_verify: bool = False
    workers: int = 1
    max_records: int | None = None

    def __post_init__(self) -> None:
        if self.is_afrivoice:
            slug = (self.language or "swahili").lower()
            if slug not in AFRIVOICE_LANGUAGES:
                known = ", ".join(sorted(AFRIVOICE_LANGUAGES))
                raise ValueError(f"Unknown Afrivoice language {self.language!r}. Expected one of: {known}")
            self.language = slug
            return

        if not self.language:
            raise ValueError("language is required when dataset_type is 'anv'")
        slug = self.language.lower()
        if slug not in ANV_LANGUAGES:
            known = ", ".join(sorted(ANV_LANGUAGES))
            raise ValueError(f"Unknown language {self.language!r}. Expected one of: {known}")
        self.language = slug

    @property
    def afrivoice_language(self) -> str:
        return self.language or "swahili"

    @property
    def is_swahili_domain_layout(self) -> bool:
        return self.is_afrivoice and self.afrivoice_layout == "domain_split"

    @property
    def afrivoice_layout(self) -> str:
        if not self.is_afrivoice:
            return ""
        return str(AFRIVOICE_LANGUAGES[self.afrivoice_language]["layout"])

    @property
    def is_anv(self) -> bool:
        return self.dataset_type == "anv"

    @property
    def is_afrivoice(self) -> bool:
        return self.dataset_type == "afrivoice"

    @property
    def language_name(self) -> str | None:
        if not self.language:
            return None
        if self.is_anv:
            return ANV_LANGUAGES[self.language]
        meta = AFRIVOICE_LANGUAGES.get(self.language, {})
        return str(meta.get("display", self.language.title()))

    # --- Afrivoice path helpers ---

    def afrivoice_language_folder(self) -> Path:
        """Resolve the on-disk folder for a flat-layout Afrivoice language."""
        root = self.dataset_root.resolve()
        if self._looks_like_afrivoice_language_folder(root):
            return root

        folder_names = AFRIVOICE_LANGUAGES[self.afrivoice_language]["folder_names"]
        for name in folder_names:
            candidate = root / str(name)
            if candidate.is_dir():
                return candidate
        if folder_names:
            return root / str(folder_names[0])
        return root

    @staticmethod
    def _looks_like_afrivoice_language_folder(path: Path) -> bool:
        if not path.is_dir():
            return False
        for pattern in MANIFEST_GLOBS:
            if any(path.glob(pattern)):
                return True
        return False

    def split_folder_name(self, domain: str, split: str) -> str:
        return f"{domain}_swahili_{split}"

    def split_folder_path(self, domain: str, split: str) -> Path:
        return self.dataset_root / self.split_folder_name(domain, split)

    # --- Anv-ke path helpers ---

    def _resolve_anv_language_folder(self, parent: Path) -> Path | None:
        if not parent.is_dir() or not self.language:
            return None

        for name in (self.language_name, self.language):
            if not name:
                continue
            candidate = parent / name
            if self._looks_like_language_root(candidate):
                return candidate

        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            if child.name.lower() == self.language and self._looks_like_language_root(child):
                return child
        return None

    def language_dataset_root(self) -> Path:
        direct = self.dataset_root.resolve()
        if self._looks_like_language_root(direct):
            return direct
        if not self.language:
            return direct

        if direct.name.lower() == self.language or direct.name == self.language_name:
            resolved = self._resolve_anv_language_folder(direct.parent)
            if resolved is not None:
                return resolved
            if self.language_name and (direct.parent / self.language_name).is_dir():
                return direct.parent / self.language_name
            return direct

        resolved = self._resolve_anv_language_folder(direct)
        if resolved is not None:
            return resolved
        if self.language_name:
            return direct / self.language_name
        return direct / self.language

    @staticmethod
    def _looks_like_language_root(path: Path) -> bool:
        if not path.is_dir():
            return False
        for split in ANV_SPLITS:
            split_dir = path / split
            if not split_dir.is_dir():
                continue
            for style in ANV_STYLES:
                if (split_dir / style / "audios").is_dir():
                    return True
        return False

    def cleaned_manifest_path(self, split: str, style: str) -> Path:
        return self.output_root / self.language / split / style / "manifest_cleaned.jsonl"

    def features_split_dir(self, split: str, style: str) -> Path:
        return self.features_dir / self.language / split / style

    @classmethod
    def with_work_dir(cls, work_dir: Path, **kwargs) -> "PipelineConfig":
        """Place all large pipeline outputs under one directory (e.g. community storage)."""
        work_dir = work_dir.resolve()
        defaults = {
            "extract_cache_root": work_dir / "extracted",
            "output_root": work_dir / "cleaned",
            "processed_root": work_dir / "processed",
            "features_dir": work_dir / "features",
            "stats_dir": work_dir / "statistics",
        }
        defaults.update(kwargs)
        return cls(**defaults)
