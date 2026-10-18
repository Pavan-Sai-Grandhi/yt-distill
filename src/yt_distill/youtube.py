"""Resolve a YouTube URL to videos and fetch their audio (yt-dlp)."""

import json
import re
from dataclasses import dataclass, field

import yt_dlp

from .paths import video_dir

_BASE = {"quiet": True, "no_warnings": True, "noprogress": True}


@dataclass
class Source:
    kind: str  # "video" | "playlist" | "channel"
    title: str
    url: str
    videos: list[dict] = field(default_factory=list)  # {id, title, duration} in source order


def _normalize(url: str) -> str:
    # A bare channel URL lists tabs, not videos; point it at the uploads tab.
    if re.search(r"youtube\.com/(@[^/?#]+|channel/[^/?#]+|c/[^/?#]+)/?$", url):
        return url.rstrip("/") + "/videos"
    return url


def resolve(url: str) -> Source:
    url = _normalize(url)
    with yt_dlp.YoutubeDL(_BASE | {"extract_flat": "in_playlist"}) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") != "playlist":
        return Source("video", info["title"], url, [{"id": info["id"], "title": info["title"], "duration": info.get("duration")}])
    videos = [
        {"id": e["id"], "title": e.get("title"), "duration": e.get("duration")}
        for e in info["entries"]
        if e and e.get("ie_key", "Youtube") == "Youtube" and e.get("id")
    ]
    kind = "playlist" if "list=" in url else "channel"
    return Source(kind, info.get("title") or info["id"], url, videos)


def fetch_audio(video_id: str) -> dict:
    """Download best audio as Opus into data/<id>/ and record metadata. Skips if present."""
    d = video_dir(video_id)
    meta_path = d / "meta.json"
    if (d / "audio.opus").exists() and meta_path.exists():
        return json.loads(meta_path.read_text())
    opts = _BASE | {
        "format": "bestaudio/best",
        "outtmpl": str(d / "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "opus"}],
        "sleep_interval_requests": 1,
        "retries": 5,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    meta = {
        "id": video_id,
        "title": info.get("title"),
        "channel": info.get("channel"),
        "channel_handle": (info.get("uploader_id") or "").lstrip("@") or info.get("channel_id"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "url": info.get("webpage_url"),
        "chapters": [{"start": c["start_time"], "title": c["title"]} for c in info.get("chapters") or []],
        "description": info.get("description"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
