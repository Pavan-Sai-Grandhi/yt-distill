#!/usr/bin/env bash
# Fetch the ASR models into models/. Idempotent; re-run any time.
#   vaani-te / vaani-hi  ARTPARK-IISc/whisper-large-v3-vaani-{telugu,hindi} (HF transformers) → ggml → q8_0,
#                        and → MLX fp16 in mlx-vaani-<lang>/ (the fast default; the ggml file is the fallback)
#   turbo                stock whisper large-v3-turbo (baseline)
#   VAD                  Silero v5 for whisper.cpp
# Pass language names to fetch only some Vaani models, e.g. `setup_models.sh hindi`.
set -euo pipefail
mkdir -p "$(dirname "$0")/../models/src"
cd "$(dirname "$0")/../models"

# Every output is written to *.part and renamed when complete, so an interrupted run never leaves a "done" file.
get() { [ -s "$2" ] || { curl -fL --progress-bar -C - -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
get https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin ggml-large-v3-turbo.bin
get https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin ggml-silero-v5.1.2.bin

for l in ${@:-telugu hindi}; do
  if [ "$l" = telugu ]; then code=te; elif [ "$l" = hindi ]; then code=hi; else echo "unknown language $l" >&2; exit 1; fi
  out=ggml-vaani-large-v3-$code-q8_0.bin mlx=mlx-vaani-$code
  [ -s "$out" ] && [ -s "$mlx/weights.safetensors" ] && continue
  uvx --from huggingface_hub hf download "ARTPARK-IISc/whisper-large-v3-vaani-$l" --exclude trainer_state.json --local-dir "src/vaani-$l"
  if [ ! -s "$out" ]; then
    [ -d src/whisper.cpp ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git src/whisper.cpp
    [ -d src/openai-whisper ] || git clone --depth 1 https://github.com/openai/whisper.git src/openai-whisper
    uv run --no-project --with torch --with transformers --with numpy \
      python src/whisper.cpp/models/convert-h5-to-ggml.py "src/vaani-$l" src/openai-whisper src >/dev/null
    whisper-quantize src/ggml-model.bin "$out.part" q8_0 >/dev/null && mv "$out.part" "$out"
  fi
  if [ ! -s "$mlx/weights.safetensors" ]; then
    rm -rf "$mlx.part"
    uv run --no-project --python 3.12 --with mlx-whisper --with safetensors \
      python -I ../src/yt_distill/mlx_convert.py "src/vaani-$l" "$mlx.part" && rm -rf "$mlx" && mv "$mlx.part" "$mlx"
  fi
  rm -rf src/ggml-model.bin "src/vaani-$l"  # ~9 GB of intermediates
done
ls -lh ./*.bin mlx-*/weights.safetensors
