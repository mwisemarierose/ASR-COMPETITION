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
    duration_sec: float | None = None

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
    *,
    require_transcript: bool = True,
) -> Iterator[TrainingRecord]:
    for domain in domains:
        manifest_path = _swahili_manifest_path(work_dir, domain, split)
        if not manifest_path.is_file():
            continue
        for row in _load_jsonl(manifest_path):
            transcript = str(row.get("transcript") or "").strip()
            audio_path = str(row.get("audio_path") or "").strip()
            if require_transcript and not transcript:
                continue
            if not audio_path:
                continue
            key = str(row.get("key") or Path(audio_path).stem)
            yield TrainingRecord(
                key=key,
                sentence=transcript,
                language="swahili",
                audio_path=audio_path,
                source=f"swahili/{domain}/{split}",
                duration_sec=_row_duration_sec(row),
            )


def iter_anv_records(
    work_dir: Path,
    split: str,
    languages: tuple[str, ...] = COMPETITION_ANV_LANGUAGES,
    *,
    skip_maasai_scripted_train: bool = True,
    require_transcript: bool = True,
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
                if require_transcript and not transcript:
                    continue
                if not isinstance(audio_source, dict):
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
                    duration_sec=_row_duration_sec(row),
                )


def collect_records(
    work_dir: Path,
    split: str,
    *,
    swahili_split: str | None = None,
    anv_split: str | None = None,
    swahili_domains: tuple[str, ...] = DOMAINS,
    anv_languages: tuple[str, ...] = COMPETITION_ANV_LANGUAGES,
    include_swahili: bool = True,
    include_anv: bool = True,
    skip_maasai_scripted_train: bool = True,
    require_transcript: bool = True,
    max_samples: int | None = None,
    max_samples_per_source: int | None = None,
    seed: int = 42,
) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    sw_split = swahili_split or split
    anv_split_name = anv_split or split

    if include_swahili:
        records.extend(
            iter_swahili_records(
                work_dir,
                sw_split,
                domains=swahili_domains,
                require_transcript=require_transcript,
            )
        )
    if include_anv:
        records.extend(
            iter_anv_records(
                work_dir,
                anv_split_name,
                languages=anv_languages,
                skip_maasai_scripted_train=skip_maasai_scripted_train,
                require_transcript=require_transcript,
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


_AUDIO_SKIP_COUNT = 0


def try_load_record_audio(
    record: TrainingRecord,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any] | None:
    """Load audio; return None on corrupt/missing clips instead of crashing training."""
    global _AUDIO_SKIP_COUNT
    try:
        return load_record_audio(record, sample_rate=sample_rate)
    except Exception as exc:
        _AUDIO_SKIP_COUNT += 1
        if _AUDIO_SKIP_COUNT <= 20 or _AUDIO_SKIP_COUNT % 100 == 0:
            print(
                f"WARNING: skipping corrupt audio ({_AUDIO_SKIP_COUNT} total) "
                f"key={record.key} source={record.source}: {exc}",
                flush=True,
            )
        return None


def audio_skip_count() -> int:
    return _AUDIO_SKIP_COUNT


def _row_duration_sec(row: dict[str, Any]) -> float | None:
    for key in ("duration_sec", "duration", "csv_duration_sec"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def summarize_record_durations(records: list[TrainingRecord]) -> dict[str, Any]:
    """Sum manifest durations for the exact training subset."""
    total_sec = 0.0
    known = 0
    missing = 0
    by_language_sec: dict[str, float] = {}
    by_language_known: dict[str, int] = {}

    for record in records:
        if record.duration_sec is None:
            missing += 1
            continue
        total_sec += record.duration_sec
        known += 1
        by_language_sec[record.language] = by_language_sec.get(record.language, 0.0) + record.duration_sec
        by_language_known[record.language] = by_language_known.get(record.language, 0) + 1

    by_language_hours = {
        language: by_language_sec[language] / 3600.0
        for language in sorted(by_language_sec)
    }
    return {
        "clips": len(records),
        "clips_with_duration": known,
        "clips_missing_duration": missing,
        "total_seconds": total_sec,
        "total_hours": total_sec / 3600.0,
        "avg_seconds_known_clips": (total_sec / known) if known else None,
        "hours_by_language": by_language_hours,
        "clips_by_language_with_duration": dict(sorted(by_language_known.items())),
    }


def format_duration_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Clips: {summary['clips']:,} "
        f"(duration known for {summary['clips_with_duration']:,}; "
        f"missing for {summary['clips_missing_duration']:,})",
        f"Total audio: {summary['total_hours']:.1f} hours "
        f"({summary['total_seconds']:,.0f} seconds)",
    ]
    if summary["avg_seconds_known_clips"] is not None:
        lines.append(f"Average clip length (known): {summary['avg_seconds_known_clips']:.2f}s")
    lines.append("Hours by language:")
    for language, hours in summary["hours_by_language"].items():
        clips = summary["clips_by_language_with_duration"].get(language, 0)
        lines.append(f"  {language}: {hours:.1f}h ({clips:,} clips)")
    return "\n".join(lines)


def subsample_eval_records(
    records: list[TrainingRecord],
    max_samples: int | None,
    seed: int = 42,
) -> list[TrainingRecord]:
    """Balanced dev subset for faster training-time eval; full dev used at end."""
    if max_samples is None or len(records) <= max_samples:
        return records

    by_language: dict[str, list[TrainingRecord]] = {}
    for record in records:
        by_language.setdefault(record.language, []).append(record)

    per_language = max(1, max_samples // len(by_language))
    rng = random.Random(seed)
    sampled: list[TrainingRecord] = []
    for language in sorted(by_language):
        rows = by_language[language]
        if len(rows) <= per_language:
            sampled.extend(rows)
        else:
            sampled.extend(rng.sample(rows, per_language))

    if len(sampled) > max_samples:
        rng.shuffle(sampled)
        sampled = sampled[:max_samples]
    return sampled


def summarize_records(records: list[TrainingRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.language] = counts.get(record.language, 0) + 1
    return dict(sorted(counts.items()))


def whisper_language_code(language: str) -> str | None:
    return WHISPER_LANGUAGE_CODES.get(language.lower())


def decoder_prompt_token_ids(processor: Any, language: str) -> list[int]:
    """Return Whisper decoder prompt token ids for a competition language."""
    lang_code = whisper_language_code(language)
    forced = processor.get_decoder_prompt_ids(language=lang_code, task="transcribe")
    return [token_id for _, token_id in forced]


def set_forced_language_prompt(model: Any, processor: Any, language: str) -> None:
    """Force language/task tokens during generation for one language."""
    lang_code = whisper_language_code(language)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=lang_code,
        task="transcribe",
    )
    model.config.suppress_tokens = []
