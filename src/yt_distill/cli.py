import argparse
import sys

from . import asr, bakeoff, pipeline

USAGE_EXAMPLES = """examples:
  yt-distill https://www.youtube.com/watch?v=VIDEO_ID
  yt-distill "https://www.youtube.com/playlist?list=PLAYLIST_ID" --limit 3
  yt-distill "https://www.youtube.com/watch?v=VIDEO_ID" --lang hi          Hindi / Hinglish
  yt-distill bakeoff prepare URL1 URL2 ...   then   yt-distill bakeoff score
"""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in {"distill", "bakeoff", "-h", "--help"}:
        argv = ["distill", *argv]  # `yt-distill <url>` is the common case

    p = argparse.ArgumentParser(prog="yt-distill", epilog=USAGE_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("distill", help="video / playlist / channel URL → transcripts + notes")
    d.add_argument("url")
    d.add_argument("--lang", default="te", help="spoken language: te (default), hi, or any Whisper code; 'auto' detects per chunk (turbo only)")
    d.add_argument("--model", choices=asr.ASR_MODELS, help="ASR model family (default: best for --lang)")
    d.add_argument("--limit", type=int, help="only the first N videos")
    d.add_argument("--transcribe-only", action="store_true", help="skip the Claude notes step")
    d.add_argument("--force", action="store_true", help="redo transcripts and notes even if cached")
    d.add_argument("--dry-run", action="store_true", help="list the videos and total hours, then stop")

    b = sub.add_parser("bakeoff", help="compare ASR models on hand-corrected clips")
    bsub = b.add_subparsers(dest="action", required=True)
    bp = bsub.add_parser("prepare")
    bp.add_argument("urls", nargs="+")
    bp.add_argument("--start", type=float, help="clip start in seconds (default 40%% into the video)")
    bp.add_argument("--length", type=float, default=60)
    bp.add_argument("--lang", default="te")
    bsub.add_parser("score")
    bs = bsub.add_parser("sheet", help="phone-friendly page to judge outputs side by side")
    bs.add_argument("--models", help="comma-separated outputs to include, e.g. vaani,indic (default: all)")

    a = p.parse_args(argv)
    if a.cmd == "distill":
        return pipeline.run(a.url, a.model, a.lang, a.limit, not a.transcribe_only, a.force, a.dry_run)
    if a.action == "prepare":
        bakeoff.prepare(a.urls, a.start, a.length, a.lang)
    elif a.action == "sheet":
        bakeoff.sheet(a.models.split(",") if a.models else None)
    else:
        bakeoff.score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
