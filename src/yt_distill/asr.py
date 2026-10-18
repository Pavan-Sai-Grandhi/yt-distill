"""Speech-to-text via whisper.cpp (Metal). Each model is a ggml file under models/."""

import contextlib
import fcntl
import json
import math
import re
import subprocess
import tempfile
import wave
from pathlib import Path

from .paths import MODELS

# Model family → ggml file per spoken language ("*" = any language, incl. "auto" per chunk).
ASR_MODELS = {
    # ARTPARK-IISc/whisper-large-v3-vaani-{telugu,hindi}, converted + q8_0 by scripts/setup_models.sh.
    # Decoded on MLX when models/mlx-vaani-<lang>/ exists (see mlx_dir), else with these files on whisper.cpp.
    "vaani": {"te": "ggml-vaani-large-v3-te-q8_0.bin", "hi": "ggml-vaani-large-v3-hi-q8_0.bin"},
    "turbo": {"*": "ggml-large-v3-turbo.bin"},  # stock Whisper
    "indic": {},  # AI4Bharat IndicConformer (not whisper.cpp; see indic_worker.py), languages in INDIC_LANGS
}
INDIC_LANGS = {"as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"}
INDIC_ENV = ["uv", "run", "--no-project", "--python", "3.12", "--with", "transformers", "--with", "onnxruntime==1.20.1",
             "--with", "onnx==1.20.1", "--with", "torch", "--with", "numpy", "python", "-I"]
MLX_ENV = ["uv", "run", "--no-project", "--python", "3.12", "--with", "mlx-whisper", "python", "-I"]
DEFAULT_MODEL = {"te": "vaani", "hi": "vaani"}  # anything else falls back to turbo
VAD_MODEL = "ggml-silero-v5.1.2.bin"


def default_model(language: str) -> str:
    return DEFAULT_MODEL.get(language, "turbo")


def model_file(name: str, language: str) -> Path | None:
    if name == "indic":
        return Path(__file__).with_name("indic_worker.py") if language in INDIC_LANGS else None
    files = ASR_MODELS[name]
    f = files.get(language) or files.get("*")
    return MODELS / f if f else None


def mlx_dir(name: str, language: str) -> Path | None:
    """The MLX copy of a Vaani model, if setup_models.sh made one: ~5× faster than whisper.cpp (mlx_worker.py)."""
    d = MODELS / f"mlx-{name}-{language}"
    return d if name == "vaani" and (d / "weights.safetensors").exists() else None


def model_path(name: str, language: str) -> Path:
    p = model_file(name, language)
    if p is None:
        options = sorted(INDIC_LANGS) if name == "indic" else ASR_MODELS[name]
        raise SystemExit(f"ASR model '{name}' has no '{language}' variant; options: {', '.join(options)}")
    if not p.exists():
        raise SystemExit(f"ASR model '{name}' ({language}) missing at {p}; run scripts/setup_models.sh")
    return p


def to_wav(src: Path, dst: Path, start: float | None = None, length: float | None = None) -> None:
    """16 kHz mono PCM, optionally a [start, start+length) slice."""
    cut = (["-ss", str(start)] if start is not None else []) + (["-t", str(length)] if length else [])
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *cut, "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
    )


# Telugu costs so many BPE tokens per second that a full 30 s Whisper window overflows the
# decoder and gets truncated, and the Vaani fine-tune loops when decoding with timestamps.
# So: VAD → pack speech into short chunks → decode each chunk independently, no timestamps.
# MLX decodes the chunks in batches (mlx_worker.py); the whisper.cpp fallback decodes one at a time with beam 3.
CHUNK_MAX_S = 5.0  # longer chunks silently drop speech: Whisper hits its 224-token cap, IndicConformer just stops
WORKERS = 1  # whisper-cli processes; one already saturates an M4's GPU, more only add model copies
EDGE_PAD_S = (0.2, 0.4)  # VAD clips word tails; without this the last word of a chunk is lost
_VAD_LINE = re.compile(r"start = ([\d.]+), end = ([\d.]+)")


@contextlib.contextmanager
def gpu_lock():
    """One transcription at a time machine-wide: concurrent jobs over-subscribe Metal and can hang the Mac."""
    with open(MODELS / ".asr.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("  waiting for another transcription to finish", flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        yield


def speech_spans(wav: Path) -> list[tuple[float, float]]:
    out = subprocess.run(
        ["whisper-vad-speech-segments", "-vm", str(MODELS / VAD_MODEL), "-f", str(wav), "-np"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [(float(a) / 100, float(b) / 100) for a, b in _VAD_LINE.findall(out)]  # centiseconds


def pack(spans: list[tuple[float, float]], max_len: float | None = None) -> list[tuple[float, float]]:
    """Merge adjacent speech spans into chunks ≤ max_len; hard-split spans that are longer."""
    max_len = max_len or CHUNK_MAX_S
    pieces = []
    for a, b in spans:
        n = math.ceil((b - a) / max_len)
        pieces += [(a + (b - a) * i / n, a + (b - a) * (i + 1) / n) for i in range(n)]
    chunks: list[tuple[float, float]] = []
    for a, b in pieces:
        if chunks and b - chunks[-1][0] <= max_len:
            chunks[-1] = (chunks[-1][0], b)
        else:
            chunks.append((a, b))
    pre, post = EDGE_PAD_S
    padded = []
    for i, (a, b) in enumerate(chunks):
        lo = (chunks[i - 1][1] + a) / 2 if i else 0.0
        hi = (b + chunks[i + 1][0]) / 2 if i + 1 < len(chunks) else b + post
        padded.append((max(a - pre, lo), min(b + post, hi)))
    return padded


def transcribe(audio: Path, model: str, language: str = "te") -> list[dict]:
    """Return [{start, end, text}] in seconds, one entry per speech chunk."""
    with gpu_lock(), tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = tmp / "a.wav"
        to_wav(audio, wav)
        chunks = pack(speech_spans(wav))
        files = []
        with wave.open(str(wav)) as w:
            rate, frames = w.getframerate(), w.readframes(w.getnframes())
            width = w.getsampwidth()
        for i, (a, b) in enumerate(chunks):
            f = tmp / f"c{i:05d}.wav"
            with wave.open(str(f), "wb") as o:
                o.setnchannels(1), o.setsampwidth(width), o.setframerate(rate)
                o.writeframes(frames[int(a * rate) * width:int(b * rate) * width])
            files.append(f)
        if model == "indic":
            texts = _worker("IndicConformer", [*INDIC_ENV, str(model_path("indic", language))], files, language)
        elif d := mlx_dir(model, language):
            texts = _worker("MLX", [*MLX_ENV, str(Path(__file__).with_name("mlx_worker.py")), str(d)], files, language)
        else:
            texts = _whisper(files, model, language)
        return [
            {"start": round(a, 2), "end": round(b, 2), "text": t}
            for (a, b), t in zip(chunks, texts) if t
        ]


def _whisper(files: list[Path], model: str, language: str) -> list[str]:
    cmd = ["whisper-cli", "-m", str(model_path(model, language)), "-l", language, "-nt", "-mc", "0", "-bs", "3", "-nf", "-otxt", "-np"]
    procs = [
        subprocess.Popen(cmd + [x for f in files[w::WORKERS] for x in ("-f", str(f))],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for w in range(min(WORKERS, len(files)))
    ]
    if any(p.wait() for p in procs):
        raise RuntimeError("whisper-cli failed")
    # whisper.cpp can split a multi-byte character across tokens; drop the broken bytes.
    raw = (f.with_suffix(".wav.txt").read_bytes().decode("utf-8", "ignore") for f in files)
    return [" ".join(t.replace("\ufffd", "").split()) for t in raw]


def _worker(name: str, cmd: list[str], files: list[Path], language: str) -> list[str]:
    """Run a worker script that takes <out.json> <lang> <chunk.wav>... and writes a JSON list of texts."""
    if not files:
        return []
    result = files[0].with_name("texts.json")
    out = subprocess.run([*cmd, str(result), language, *map(str, files)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{name} failed: {out.stderr.strip()[-500:]}")
    return [" ".join(t.split()) for t in json.loads(result.read_text())]


def quality(segments: list[dict], duration: float | None) -> dict:
    """Cheap hallucination signals: repeated segments and text density."""
    texts = [s["text"] for s in segments]
    repeats = sum(1 for a, b in zip(texts, texts[1:]) if a == b)
    chars = sum(len(t) for t in texts)
    minutes = (duration or (segments[-1]["end"] if segments else 0)) / 60 or 1
    q = {"segments": len(texts), "repeat_ratio": round(repeats / max(len(texts), 1), 3), "chars_per_min": round(chars / minutes)}
    q["suspect"] = q["repeat_ratio"] > 0.1 or q["chars_per_min"] < 150
    return q


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def as_text(segments: list[dict]) -> str:
    return "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in segments)
