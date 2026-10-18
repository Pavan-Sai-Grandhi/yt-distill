"""Vaani on MLX: batched greedy decoding of the speech chunks, ~5× faster than whisper.cpp on an M4.

Run in its own environment (see asr.MLX_ENV) so the main package stays dependency-light:
    python mlx_worker.py <model_dir> <out.json> <lang> <chunk.wav>...  →  JSON list of texts in out.json
A decode step costs about the same for 1 row as for 16, so decoding 16 chunks at once is the win. The encoder
gets a 10 s window instead of Whisper's 30 s (chunks are ≤ 5.6 s), a third of the encoder and cross-attention work.
"""

import json
import sys
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_whisper import audio, decoding, whisper

BATCH = 16  # ~5.4 GB peak on large-v3
WINDOW_S = 10  # encoder positions are fixed sinusoids, so a shorter window is the first 500 of them


def load(model_dir: Path) -> whisper.Whisper:
    dims = json.loads((model_dir / "config.json").read_text()) | {"n_audio_ctx": WINDOW_S * 50}
    model = whisper.Whisper(whisper.ModelDimensions(**dims), mx.float16)
    model.load_weights(str(model_dir / "weights.safetensors"), strict=False)  # alignment_heads isn't a weight
    mx.eval(model.parameters())
    return model


def read(path: str) -> mx.array:
    with wave.open(path) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return audio.pad_or_trim(mx.array(pcm), WINDOW_S * audio.SAMPLE_RATE)


def main() -> None:
    model_dir, out, lang, paths = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
    model = load(model_dir)
    opts = decoding.DecodingOptions(language=lang, without_timestamps=True, temperature=0.0, fp16=True)
    texts = [""] * len(paths)
    order = sorted(range(len(paths)), key=lambda i: Path(paths[i]).stat().st_size)  # similar lengths decode together
    for k in range(0, len(order), BATCH):
        batch = order[k:k + BATCH]
        mel = mx.stack([audio.log_mel_spectrogram(read(paths[i]), n_mels=model.dims.n_mels)[:WINDOW_S * 100] for i in batch])
        for i, r in zip(batch, decoding.decode(model, mel.astype(mx.float16), opts)):
            texts[i] = r.text.replace("�", "")  # a multi-byte character split across tokens
        mx.clear_cache()  # MLX keeps freed buffers: ~9 GB on top of the ~5 GB in use, enough to push a 16 GB Mac into swap
    with open(out, "w") as f:
        json.dump(texts, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
