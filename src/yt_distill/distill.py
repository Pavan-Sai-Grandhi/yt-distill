"""Turn transcripts into notes with headless Claude Code (`claude -p`, uses the subscription)."""

import os
import re
import subprocess
from pathlib import Path

from .asr import fmt_ts
from .paths import GLOSSARY, KB, PROMPTS

CLAUDE_MODEL = os.environ.get("YT_DISTILL_CLAUDE_MODEL", "sonnet")
LANGUAGES = {"te": "Telugu", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam"}


def _fill(template: str, **values: str) -> str:
    # One pass, so braces inside a transcript are never treated as placeholders.
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)


def claude(prompt: str) -> str:
    out = subprocess.run(
        [
            "claude", "-p", "--model", CLAUDE_MODEL, "--tools", "", "--output-format", "text",
            "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
        ],
        input=prompt, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        # Usage-limit and auth messages arrive on stdout, not stderr.
        raise RuntimeError(f"claude -p failed ({out.returncode}): {(out.stderr or out.stdout).strip()[:500]}")
    return out.stdout.strip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "untitled"


def note_path(meta: dict) -> Path:
    channel = _slug(meta.get("channel_handle") or meta.get("channel") or "unknown")
    return KB / "videos" / channel / f"{meta.get('upload_date') or 'undated'}-{meta['id']}.md"


MIXED_BATCH_LINES = 150  # keeps each rewrite well inside the output-token limit


def mixed_script(transcript: str, language: str) -> str:
    """Rewrite English words from native script into Latin (Tinglish/Hinglish), line for line."""
    lines = transcript.splitlines()
    out = []
    for i in range(0, len(lines), MIXED_BATCH_LINES):
        batch = lines[i:i + MIXED_BATCH_LINES]
        rewritten = claude(_fill(
            (PROMPTS / "mixed-script.md").read_text(),
            language=LANGUAGES.get(language, language),
            glossary=GLOSSARY.read_text() if GLOSSARY.exists() else "(none)",
            transcript="\n".join(batch),
        )).splitlines()
        rewritten = [l for l in rewritten if l.strip()]
        # Never lose speech to a rewrite: if lines went missing, keep the original batch.
        out += rewritten if len(rewritten) >= len(batch) * 0.95 else batch
    return "\n".join(out)


def distill_video(meta: dict, transcript: str, language: str) -> Path:
    chapters = "".join(f"- [{fmt_ts(c['start'])}] {c['title']}\n" for c in meta.get("chapters", []))
    prompt = _fill(
        (PROMPTS / "distill-video.md").read_text(),
        language=LANGUAGES.get(language, language),
        title=meta["title"], channel=meta.get("channel") or "", url=meta["url"],
        upload_date=meta.get("upload_date") or "unknown",
        chapters=f"Chapters:\n{chapters}" if chapters else "",
        glossary=GLOSSARY.read_text() if GLOSSARY.exists() else "(none)",
        transcript=transcript,
    )
    body = claude(prompt)
    path = note_path(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"---\nvideo_id: {meta['id']}\nurl: {meta['url']}\nchannel: {meta.get('channel')}\n"
        f"upload_date: {meta.get('upload_date')}\n---\n\n# {meta['title']}\n\n"
    )
    path.write_text(header + body + "\n")
    return path


def synthesize(kind: str, title: str, notes: list[Path], language: str, channel: str) -> Path:
    joined = "\n\n".join(f"=== Video {i}: {p.read_text()}" for i, p in enumerate(notes, 1))
    prompt = _fill(
        (PROMPTS / "synthesize-playlist.md").read_text(),
        kind=kind, title=title, language=LANGUAGES.get(language, language), notes=joined,
    )
    body = claude(prompt)
    path = KB / "playlists" / _slug(channel) / f"{_slug(title)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    index = "\n".join(f"{i}. [{_title_of(p)}](../../videos/{p.parent.name}/{p.name})" for i, p in enumerate(notes, 1))
    path.write_text(f"# {title}\n\n{body}\n\n## Videos\n\n{index}\n")
    return path


def _title_of(note: Path) -> str:
    m = re.search(r"^# (.+)$", note.read_text(), re.M)
    return m.group(1) if m else note.stem
