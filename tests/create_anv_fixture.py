"""Build a tiny Anv-ke-style fixture for local pipeline smoke tests."""
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "anv_kalenjin"


def _wav_bytes(duration_sec: float = 3.0, sample_rate: int = 16_000) -> bytes:
    sample_count = int(duration_sec * sample_rate)
    audio = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sample_count) / sample_rate)).astype(np.float32)
    pcm = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def build_fixture(root: Path = FIXTURE_ROOT) -> Path:
    split_dir = root / "dev" / "unscripted"
    audio_dir = split_dir / "audios"
    files_dir = split_dir / "files"
    audio_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "recorder_uuid": "speaker-001",
            "transcription": "Chamgei [pause] akobo.",
            "audio": {"bytes": _wav_bytes(3.0), "path": "clip-001.wav"},
        },
        {
            "recorder_uuid": "speaker-002",
            "transcription": "Emet ab gaa.",
            "audio": {"bytes": _wav_bytes(2.5), "path": "clip-002.wav"},
        },
    ]

    table = pa.table(
        {
            "recorder_uuid": [row["recorder_uuid"] for row in rows],
            "transcription": [row["transcription"] for row in rows],
            "audio": [row["audio"] for row in rows],
        }
    )
    pq.write_table(table, audio_dir / "dev_unscripted_000.parquet")

    (files_dir / "meta.csv").write_text(
        "recorder_uuid,domain,dialect,duration,language,type\n"
        "speaker-001,health,NANDI,3.0,Kalenjin,unscripted\n"
        "speaker-002,agriculture,KIPSIGIS,2.5,Kalenjin,unscripted\n",
        encoding="utf-8",
    )
    (files_dir / "transcripts.csv").write_text(
        "recorder_uuid,transcript,duration\n"
        "speaker-001,Chamgei [pause] akobo.,3.0\n"
        "speaker-002,Emet ab gaa.,2.5\n",
        encoding="utf-8",
    )
    return root


if __name__ == "__main__":
    path = build_fixture()
    print(f"Wrote Anv-ke fixture to {path}")
