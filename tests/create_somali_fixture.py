"""Build a tiny Afrivoice Somali-style flat folder fixture."""
from __future__ import annotations

import json
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "somali_flat"


def build_fixture(root: Path = FIXTURE_ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "audio_shards").mkdir(exist_ok=True)

    rows = [
        {
            "speaker_id": "speaker-001",
            "audio_path": "clip-001.wav",
            "transcription": "Subax wanaagsan, tani waa duubis wanaagsan.",
            "duration": 4.5,
            "locale": "so_SO",
            "gender": "Female",
            "year": "2024",
        },
        {
            "speaker_id": "speaker-002",
            "audio_path": "clip-002.wav",
            "transcription": "Maalin wanaagsan.",
            "duration": 3.0,
            "locale": "so_SO",
            "gender": "Male",
            "year": "2024",
        },
    ]

    manifest = root / "manifest_0.json"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return root


if __name__ == "__main__":
    path = build_fixture()
    print(f"Wrote Somali-style fixture to {path}")
