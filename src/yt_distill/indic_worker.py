"""IndicConformer (AI4Bharat, gated: accept the licence on Hugging Face and `hf auth login`).

Run in its own environment (see asr.INDIC_ENV) so the main package stays dependency-light:
    python indic_worker.py <out.json> <lang> <chunk.wav>...  →  JSON list of texts in out.json
(a file, not stdout: the model's native code prints to the process's stdout)
"""

import json
import sys
import wave

import numpy as np
import torch
from transformers import AutoModel

REPO = "ai4bharat/indic-conformer-600m-multilingual"


def main() -> None:
    out, lang, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True)
    texts = []
    for p in paths:
        with wave.open(p) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        texts.append(model(torch.from_numpy(audio)[None], lang, "rnnt"))  # RNNT: ~10x realtime, fuller than CTC
    with open(out, "w") as f:
        json.dump(texts, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
