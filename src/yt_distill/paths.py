"""Project locations. YT_DISTILL_HOME overrides the root (defaults to the repo checkout)."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("YT_DISTILL_HOME", Path(__file__).resolve().parents[2]))
MODELS = ROOT / "models"
DATA = ROOT / "data"  # data/<video_id>/{meta.json, audio.opus, transcript.<model>.json}
KB = ROOT / "kb"  # kb/videos/<channel>/..., kb/playlists/...
PROMPTS = ROOT / "prompts"
GLOSSARY = ROOT / "glossary.md"


def video_dir(video_id: str) -> Path:
    d = DATA / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d
