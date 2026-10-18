# yt-distill

Turn YouTube videos that have **no subtitles**, in Telugu, Hindi/Hinglish or another Indian language, into English study notes. It runs locally and free: `yt-dlp` → Telugu speech recognition with Whisper on the Mac's GPU (MLX, or `whisper.cpp` as the fallback) → notes written by headless Claude Code on your subscription.

## Use it

From anywhere in Claude Code (including the phone app over Remote Control):

```
/distill https://www.youtube.com/watch?v=VIDEO_ID
/distill https://www.youtube.com/playlist?list=PLAYLIST_ID
```

Or from a terminal:

```bash
yt-distill "https://www.youtube.com/playlist?list=PLAYLIST_ID" --dry-run   # count videos and hours
yt-distill "https://www.youtube.com/playlist?list=PLAYLIST_ID" --limit 3   # first 3 videos
yt-distill "https://www.youtube.com/@channel"                             # a whole channel's uploads
yt-distill "https://www.youtube.com/watch?v=VIDEO_ID" --lang hi           # Hindi / Hinglish
```

Flags: `--lang` (spoken language: `te` default, `hi`, any Whisper code, or `auto` = per-chunk detection with turbo), `--model` (override the per-language default), `--transcribe-only` (skip the notes), `--force` (redo cached work).

**Languages:** the ASR model is chosen per language (`DEFAULT_MODEL` in `asr.py`): Telugu → Vaani Telugu, Hindi → see *Measured*, anything else → stock Whisper turbo. The language is never auto-detected for a whole video: Whisper's language ID labels code-mixed Telugu as English or Hindi, so pass `--lang` (`/distill` infers it from the channel and title script).

Output:

| What | Where |
|---|---|
| Per-video notes (English, Telugu terms in brackets, `[mm:ss]` citations) | `kb/videos/<channel>/<date>-<id>.md` |
| Playlist or channel course notes | `kb/playlists/<channel>/<title>.md` |
| Timestamped transcript (native script) | `data/<id>/transcript.<lang>.<model>.md` |
| Same, with English in English (Tinglish/Hinglish; fed to the notes) | `data/<id>/transcript.<lang>.<model>.mixed.md` |
| Audio (Opus, ~45 MB/h) | `data/<id>/audio.opus` |

Every stage is cached by video ID, so re-running a URL only does the missing work. Only one transcription runs at a time on the machine (a lock in `models/.asr.lock`); a second run waits its turn. Several concurrent GPU jobs once froze a 16 GB Mac. The [glossary](glossary.md) of finance terms is fed into the notes prompt; add misheard terms to it as you find them.

## Setup

```bash
brew install whisper-cpp ffmpeg
./scripts/setup_models.sh     # Telugu + Hindi (or pass `telugu` / `hindi`); ~6 GB download each, converted for MLX and whisper.cpp
uv tool install -e .          # puts `yt-distill` on PATH
```

The `/distill` skill lives in `skills/distill/SKILL.md`; install it with `ln -s "$PWD/skills/distill" ~/.claude/skills/distill`.

## How the speech recognition works, and why

- **Models:** [ARTPARK-IISc/whisper-large-v3-vaani-telugu](https://huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-telugu) and [-hindi](https://huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-hindi) (MIT), converted to MLX fp16 (`models/mlx-vaani-<lang>/`, `mlx_convert.py`) and to ggml q8_0 for the `whisper.cpp` fallback, used when the MLX copy is missing.
- **Chunking:** stock long-form Whisper decoding fails on Telugu in two ways. Telugu costs about one token per character, so a long window overflows the 224-token decode cap, and this fine-tuned model loops when timestamps are on. So `asr.py` runs Silero VAD, packs the speech into chunks of ≤5 s padded at the edges (VAD clips word endings otherwise), and decodes each chunk on its own: no timestamps, no context carried over.
- **Decoding:** `mlx_worker.py` (in its own `uv` environment with `mlx-whisper`) decodes 16 chunks at a time, greedy, with a 10 s encoder window instead of Whisper's 30 s. A decode step costs about the same for 1 row as for 16, and chunks are ≤5.6 s, so most of the 30 s window was padding. Peak memory is ~5.4 GB; the worker empties MLX's buffer cache after every batch, which otherwise grows by ~9 GB. The `whisper.cpp` fallback decodes one chunk at a time with beam 3 in a single `whisper-cli` process; one process already saturates an M4's GPU.
- **English in English:** the Vaani models write English words in Telugu/Hindi script. After transcription, a `claude -p` pass (`prompts/mixed-script.md`, 150 lines per call) rewrites them into Latin script line for line. If a rewrite comes back with lines missing, the original lines are kept.
- **IndicConformer** (`--model indic`, [AI4Bharat](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), gated, MIT): an RNNT/CTC model for 22 Indian languages, run by `indic_worker.py` in its own `uv` environment (pinned `onnxruntime==1.20.1`; newer versions reject the Hugging Face cache layout). It uses the same 5 s chunks: longer inputs silently drop speech, and a 60 s input comes back empty.
- **Quality guard:** transcripts that repeat themselves or have too little text are flagged `⚠ suspect`.
- **Speed:** see *Measured* below. Decoding dominates: Telugu costs ~1.8 byte-level tokens per character, so a 5 s chunk legitimately needs ~190 decode steps. Batching chunks is what makes this cheap.

## Measured (M4, 16 GB)

| Run | Result |
|---|---|
| Stock Whisper large-v3-turbo on Telugu | Unusable: wrong-language gibberish and repeat loops ("బట్ బట్ బట్…") |
| Vaani, whisper.cpp default long-form decoding | First 30 s fine, then degenerates into garbage bytes |
| Vaani, 12 s chunks | Readable, but drops **~half** the speech (1,183 vs 2,474 chars on the same 2-min clip) |
| Vaani, 5 s chunks, `whisper.cpp` beam 3, 3 workers (default until 2026-10-18) | Full coverage; **~0.7× real time** (a 33-min video ≈ 45 min) |
| IndicConformer, 5 s chunks (Telugu / Hindi) | 1,191 / 1,000 chars/min vs Vaani's 1,218 / 1,044 (98% / 96%) at **9.8× / 8.6× real time** |
| IndicConformer, 15 s chunks | Drops ~60% of Telugu speech (925 vs 2,225 chars on a 2-min clip) |
| Vaani, per-chunk token cap (libwhisper `max_tokens`) | No gain: beams end with a real end-of-text at ~190 tokens, just under the cap; tighter caps cut speech |
| Vaani, greedy first, beam 3 only for loops; 1 worker | 1.09× (458 vs 498 s on the 7 bake-off minutes), 99.7% of the characters |
| Vaani, `audio_ctx` 512 / 384 (shorter encoder window) | ~15–20% faster but the text changes (similarity to today's output 0.94 vs 0.97 run-to-run) |
| Vaani, 1 vs 3 `whisper-cli` workers | 1 is as fast or faster (99 vs 106 s on a Telugu minute): one process already saturates the M4 GPU |
| Vaani in HF transformers, fp16 on MPS | 34 s per 5 s chunk; batching scales linearly. Unusable |
| Vaani in MLX, batched greedy (8 chunks), 30 s window | 1.67× on Telugu (195 vs 327 s), 100.4% of the characters |
| Vaani in MLX, batched greedy (16 chunks), 10 s window (shipped) | **4.7× / 5.0×** on the Telugu / Hindi bake-off minutes (70 vs 327 s, 34 vs 171 s), 101.4% / 98.5% of the characters. Whole videos: 13.8 min Telugu in 5.0 min (**2.8× real time**, was 0.7×), 14.4 min Hindi in 4.0 min (**3.6×**, was 1.0×), both 100.3% of the characters; ~5.4 GB peak |
| Notes (`claude -p`, Sonnet) | ~1–2 min per video. See `kb/videos/daytradertelugu/20190209-82SqXcKaadU.md` |

On `whisper.cpp`, Vaani is slow because Whisper large-v3 decodes one token per step (~20 ms on an M4, whatever the beam count) and Telugu needs ~38 tokens per second of speech; the encoder also processes a full 30 s window for every 5 s chunk. MLX batching and the 10 s window remove most of both. Blind bake-off, 2026-10-16 (`bakeoff/judgements.json`): Vaani beat stock turbo on all 4 Telugu clips (1–5 wrong words per ~170); for Hindi, turbo's Hinglish read better, which the English-in-English pass now gives Vaani too. Round 2, 2026-10-17, Vaani vs IndicConformer on Telugu: IndicConformer spelled slightly better on clean teaching speech (won 2 of 4), but **dropped whole sentences** on the faster, English-heavy clips. Missing speech can't be recovered downstream and misspellings can, so **Vaani stays the default** and `--model indic` is the opt-in fast mode for large jobs. Round 3, 2026-10-18, Vaani `whisper.cpp` beam 3 vs MLX greedy with the 10 s window, Telugu: "almost same" on all 4 clips, with 28 vs 35 wrong words in ~690 (4.1% vs 5.1%). For ~5× the speed that's worth it, so MLX is now the default path.

## Bake-off: choosing the ASR model

```bash
yt-distill bakeoff prepare URL1 URL2 URL3     # 1-min clip from each video, run every installed model
# listen to bakeoff/<clip>/clip.wav, fix bakeoff/<clip>/reference.txt, set its first line to "corrected"
yt-distill bakeoff score                      # CER per model → bakeoff/report.md
yt-distill bakeoff sheet [--models vaani,indic]   # quicker: blind side-by-side page; pick the closest, click wrong words, Send
```

To add a whisper.cpp model, put its file in `models/` and add it to `ASR_MODELS` in `src/yt_distill/asr.py`.
