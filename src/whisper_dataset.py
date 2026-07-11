"""
Build HuggingFace datasets for Whisper fine-tuning from pipeline manifests.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .audio_utils import load_audio_mono, load_parquet_row_audio
from .config import ANV_STYLES, DOMAINS, SAMPLE_RATE

# Whisper tokenizer language codes (None = no dedicated token; model still learns from text).
WHISPER_LANGUAGE_CODES: dict[str, str | None] = {
    "swahili": "sw",
    "somali": "so",
    "kalenjin": None,
    "kikuyu": None,
    "dholuo": None,
    "luo": None,
    "maasai": None,
}

COMPETITION_ANV_LANGUAGES = ("kalenjin", "kikuyu", "dholuo", "somali", "maasai")


@dataclass(frozen=True)
class TrainingRecord:
    """One audio-transcript pair for Whisper fine-tuning."""

    key: str
    sentence: str
    language: str
    audio_path: str | None = None
    audio_source: dict[str, Any] | None = None
    source: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sentence": self.sentence,
            "language": self.language,
            "audio_path": self.audio_path or "",
            "audio_source": json.dumps(self.audio_source) if self.audio_source else "",
            "source": self.source,
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _swahili_manifest_path(work_dir: Path, domain: str, split: str) -> Path:
    return work_dir / "processed" / domain / split / "manifest_processed.jsonl"


def _anv_manifest_path(work_dir: Path, language: str, split: str, style: str) -> Path:
    return work_dir / "cleaned" / language / split / style / "manifest_cleaned.jsonl"


def iter_swahili_records(
    work_dir: Path,
    split: str,
    domains: tuple[str, ...] = DOMAINS,
) -> Iterator[TrainingRecord]:
    for domain in domains:
        manifest_path = _swahili_manifest_path(work_dir, domain, split)
        if not manifest_path.is_file():
            continue
        for row in _load_jsonl(manifest_path):
            transcript = str(row.get("transcript") or "").strip()
            audio_path = str(row.get("audio_path") or "").strip()
            if not transcript or not audio_path:
                continue
            key = str(row.get("key") or Path(audio_path).stem)
            yield TrainingRecord(
                key=key,
                sentence=transcript,
                language="swahili",
                audio_path=audio_path,
                source=f"swahili/{domain}/{split}",
            )


def iter_anv_records(
    work_dir: Path,
    split: str,
    languages: tuple[str, ...] = COMPETITION_ANV_LANGUAGES,
    *,
    skip_maasai_scripted_train: bool = True,
) -> Iterator[TrainingRecord]:
    for language in languages:
        for style in ANV_STYLES:
            if skip_maasai_scripted_train and language == "maasai" and split == "train" and style == "scripted":
                continue
            manifest_path = _anv_manifest_path(work_dir, language, split, style)
            if not manifest_path.is_file():
                continue
            for row in _load_jsonl(manifest_path):
                transcript = str(row.get("transcript") or "").strip()
                audio_source = row.get("audio_source")
                if not transcript or not isinstance(audio_source, dict):
                    continue
                if audio_source.get("type") != "parquet":
                    continue
                key = str(row.get("key") or f"{language}_{style}_{len(transcript)}")
                yield TrainingRecord(
                    key=key,
                    sentence=transcript,
                    language=language,
                    audio_source=audio_source,
                    source=f"{language}/{split}/{style}",
                )


def collect_records(
    work_dir: Path,
    split: str,
    *,
    swahili_domains: tuple[str, ...] = DOMAINS,
    anv_languages: tuple[str, ...] = COMPETITION_ANV_LANGUAGES,
    include_swahili: bool = True,
    include_anv: bool = True,
    skip_maasai_scripted_train: bool = True,
    max_samples: int | None = None,
    max_samples_per_source: int | None = None,
    seed: int = 42,
) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []

    if include_swahili:
        records.extend(iter_swahili_records(work_dir, split, domains=swahili_domains))
    if include_anv:
        records.extend(
            iter_anv_records(
                work_dir,
                split,
                languages=anv_languages,
                skip_maasai_scripted_train=skip_maasai_scripted_train,
            )
        )

    if max_samples_per_source is not None:
        by_source: dict[str, list[TrainingRecord]] = {}
        for record in records:
            by_source.setdefault(record.source, []).append(record)
        capped: list[TrainingRecord] = []
        rng = random.Random(seed)
        for source_rows in by_source.values():
            if len(source_rows) <= max_samples_per_source:
                capped.extend(source_rows)
            else:
                capped.extend(rng.sample(source_rows, max_samples_per_source))
        records = capped

    if max_samples is not None and len(records) > max_samples:
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)

    return records


def balance_records_by_language(
    records: list[TrainingRecord],
    mode: str = "equal",
    max_per_language: int | None = None,
    seed: int = 42,
) -> list[TrainingRecord]:
    """
    Rebalance train data so no single language dominates mixed batches.

    - equal: use the same count from every language (smallest language size)
    - cap: use at most max_per_language clips per language
    - none: return records unchanged
    """
    if mode == "none" or not records:
        return records

    by_language: dict[str, list[TrainingRecord]] = {}
    for record in records:
        by_language.setdefault(record.language, []).append(record)

    if mode == "equal":
        cap = min(len(rows) for rows in by_language.values())
    elif mode == "cap":
        if max_per_language is None:
            raise ValueError("max_per_language is required when balance_languages=cap")
        cap = max_per_language
    else:
        raise ValueError(f"Unknown balance_languages mode: {mode}")

    rng = random.Random(seed)
    balanced: list[TrainingRecord] = []
    for language in sorted(by_language):
        rows = by_language[language]
        if len(rows) <= cap:
            balanced.extend(rows)
        else:
            balanced.extend(rng.sample(rows, cap))
    rng.shuffle(balanced)
    return balanced


def load_record_audio(record: TrainingRecord, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    if record.audio_path:
        array = load_audio_mono(Path(record.audio_path), sample_rate=sample_rate)
    elif record.audio_source:
        array = load_parquet_row_audio(
            Path(record.audio_source["path"]),
            int(record.audio_source["row"]),
            sample_rate=sample_rate,
        )
    else:
        raise RuntimeError(f"No audio reference for record {record.key}")

    return {"array": array, "sampling_rate": sample_rate}


def summarize_records(records: list[TrainingRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.language] = counts.get(record.language, 0) + 1
    return dict(sorted(counts.items()))


def whisper_language_code(language: str) -> str | None:
    return WHISPER_LANGUAGE_CODES.get(language.lower())
