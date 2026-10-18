---
name: distill
description: Distill a YouTube video, playlist or channel URL into English study notes, even when it has no subtitles and is in Telugu, Hindi/Hinglish or another Indian language. Use when the user says "/distill <url>", "distill this video/playlist", or pastes a YouTube link asking for notes or a summary of it. Runs the local yt-distill pipeline (yt-dlp → Whisper ASR on MLX, whisper.cpp fallback → Claude notes).
---

# /distill <url> [extra flags]

Wraps the `yt-distill` CLI (project: `~/personal/play/yt-distill`, README there). Outputs:
- per-video notes: `kb/videos/<channel>/<date>-<id>.md`
- playlist/channel course notes: `kb/playlists/<channel>/<title>.md`
- transcripts: `data/<id>/transcript.<lang>.<model>.md` (timestamped, original language)

## Steps

1. **Size the job:** `yt-distill "<url>" --dry-run`. Report the video count and hours.
   - A single video, or less than ~3h of audio: go ahead.
   - Bigger: ask the user whether to cap it with `--limit N`, or to use the fast mode `--model indic` (~10× real time, ~3× the default's speed, but can drop sentences on fast or English-heavy speech; the default Vaani is ~3× real time and complete). Note-writing uses the Claude subscription.
2. **Pick `--lang`** (the speech model depends on it; automatic detection mislabels code-mixed speech, so the CLI never guesses):
   - The user said a language → use it.
   - Known channels: DAY TRADER తెలుగు, Money Purse → `te`; SOIC → `hi`.
   - Titles in Telugu script → `te`; in Devanagari → `hi`.
   - Otherwise (English titles from an unknown channel) → ask the user.
3. **Run** `yt-distill "<url>" --lang <code> [--limit N]` with Bash `run_in_background: true` (long-running; you are notified on exit). Pass through any flags the user gave (`--transcribe-only`, `--force`, `--model indic` for ~10x faster IndicConformer, `--model turbo`, `--lang auto`).
4. **On completion:** read the run output. For each `→ path` note, open it and give the user:
   - a 3–5 line summary of the most useful points,
   - the note paths as links,
   - any `⚠ suspect transcript` or `FAILED` lines, with a suggested fix (re-run with `--force`, or `--model turbo` to compare).
5. If the user is on another device, send the playlist or video note with SendUserFile.

Never re-download or re-transcribe on purpose: every stage is cached by video ID, so re-running the same URL only fills in what is missing.
