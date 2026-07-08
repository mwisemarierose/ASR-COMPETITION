"""
Project settings and folder paths for the Afrivoice_Swahili cleaning pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Domains in DigitalUmuganda/Afrivoice_Swahili
DOMAINS = ("agriculture", "education", "financial", "government", "health")
SPLITS = ("train", "dev", "test")

# Default raw dataset location — override with --dataset-root or DATASET_ROOT env var
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("DATASET_ROOT", PROJECT_ROOT / "data" / "raw" / "Afrivoice_Swahili")
)

CLEANED_ROOT = PROJECT_ROOT / "data" / "cleaned"
EXTRACTED_ROOT = PROJECT_ROOT / "data" / "extracted"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
STATS_DIR = PROJECT_ROOT / "outputs" / "statistics"
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"

# Cleaning thresholds (requirement: remove audio < 2 seconds)
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = None  # no upper limit — keep clips longer than 60s
ALIGNMENT_DURATION_TOLERANCE_SEC = 2.0
MIN_CHARS_PER_SEC = 2.0
MAX_CHARS_PER_SEC = 30.0

# Preprocessing + features (requirement: 16 kHz, 80 mel bins)
SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160

# Manifest + audio layout inside each split folder (e.g. agriculture_swahili_dev/)
AUDIO_DIRNAME = "audio"
MANIFEST_GLOB = "manifest_*.jsonl"

# Afrivoice stores extracted audio as hash-like .webm files, e.g. MsYzAS3P092NBPdEBTMa.webm
AUDIO_EXTENSIONS = (".webm", ".wav", ".mp3", ".ogg", ".m4a")
DEFAULT_AUDIO_EXTENSION = ".webm"


@dataclass
class PipelineConfig:
    """Runtime configuration for a cleaning run."""

    dataset_root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT)
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

    def split_folder_name(self, domain: str, split: str) -> str:
        return f"{domain}_swahili_{split}"

    def split_folder_path(self, domain: str, split: str) -> Path:
        return self.dataset_root / self.split_folder_name(domain, split)

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
