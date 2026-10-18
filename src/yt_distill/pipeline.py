"""URL → audio → transcript → notes (→ playlist course notes). Every stage is cached per video."""

import json
import time
from pathlib import Path

from . import asr, distill, youtube
from .paths import video_dir


def transcript_for(video_id: str, model: str, language: str, force: bool = False) -> tuple[list[dict], dict]:
    d = video_dir(video_id)
    cache = d / f"transcript.{language}.{model}.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return data["segments"], data["quality"]
    meta = json.loads((d / "meta.json").read_text())
    t0 = time.monotonic()
    segments = asr.transcribe(d / "audio.opus", model, language)
    quality = asr.quality(segments, meta.get("duration")) | {
        "rtf": round((meta.get("duration") or 0) / max(time.monotonic() - t0, 1e-6), 1)
    }
    cache.write_text(json.dumps({"model": model, "quality": quality, "segments": segments}, ensure_ascii=False, indent=1))
    (d / f"transcript.{language}.{model}.md").write_text(asr.as_text(segments) + "\n")
    return segments, quality


def mixed_for(video_id: str, model: str, language: str, segments: list[dict], force: bool) -> str:
    """Code-mixed (Tinglish/Hinglish) transcript; cached next to the native-script one."""
    path = video_dir(video_id) / f"transcript.{language}.{model}.mixed.md"
    if path.exists() and not force:
        return path.read_text()
    text = distill.mixed_script(asr.as_text(segments), language)
    path.write_text(text + "\n")
    return text


def run(url: str, model: str | None, language: str, limit: int | None, notes: bool, force: bool, dry_run: bool = False) -> int:
    model = model or asr.default_model(language)
    asr.model_path(model, language)  # fail fast, before any downloads
    src = youtube.resolve(url)
    videos = src.videos[:limit] if limit else src.videos
    hours = sum(v.get("duration") or 0 for v in videos) / 3600
    print(f"{src.kind}: {src.title} — {len(videos)} video(s), {hours:.1f}h of audio")
    if dry_run:
        for v in videos:
            print(f"  {v['id']}  {(v.get('duration') or 0) / 60:5.1f} min  {v.get('title')}")
        return 0

    done: list[Path] = []
    failed: list[tuple[str, str]] = []
    channel = ""
    for i, v in enumerate(videos, 1):
        tag = f"[{i}/{len(videos)}] {v['id']} {v.get('title') or ''}"[:100]
        try:
            print(f"{tag}\n  fetch", flush=True)
            meta = youtube.fetch_audio(v["id"])
            channel = channel or meta.get("channel_handle") or meta.get("channel") or ""
            print("  transcribe", flush=True)
            segments, q = transcript_for(v["id"], model, language, force)
            print(f"  {q['segments']} segments, {q['chars_per_min']} chars/min, {q['rtf']}x realtime"
                  + ("  ⚠ suspect transcript" if q["suspect"] else ""), flush=True)
            if not notes:
                continue
            transcript = asr.as_text(segments)
            if language not in ("en", "auto"):
                print("  english → latin script", flush=True)
                transcript = mixed_for(v["id"], model, language, segments, force)
            note = distill.note_path(meta)
            if note.exists() and not force:
                print(f"  note exists: {note}")
            else:
                print("  distill", flush=True)
                note = distill.distill_video(meta, transcript, language)
                print(f"  → {note}")
            done.append(note)
        except Exception as e:  # keep going; one bad video shouldn't sink a playlist
            failed.append((v["id"], str(e)))
            print(f"  ✗ {e}", flush=True)

    if notes and src.kind != "video" and len(done) > 1:
        print("synthesize", flush=True)
        print(f"  → {distill.synthesize(src.kind, src.title, done, language, channel)}")
    for vid, err in failed:
        print(f"FAILED {vid}: {err}")
    return 1 if failed else 0
