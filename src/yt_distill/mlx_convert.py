"""Convert a Hugging Face Whisper checkpoint into an mlx_whisper model directory (config.json + fp16 weights).

Used by scripts/setup_models.sh, in the asr.MLX_ENV environment plus safetensors:
    python mlx_convert.py <hf_dir> <out_dir>
"""

import json
import re
import sys
from pathlib import Path

RENAMES = [
    (r"^model\.", ""), (r"\.layers\.", ".blocks."), (r"\.self_attn\.", ".attn."), (r"\.encoder_attn\.", ".cross_attn."),
    (r"\.q_proj\.", ".query."), (r"\.k_proj\.", ".key."), (r"\.v_proj\.", ".value."), (r"\.out_proj\.", ".out."),
    (r"\.self_attn_layer_norm\.", ".attn_ln."), (r"\.encoder_attn_layer_norm\.", ".cross_attn_ln."),
    (r"\.fc1\.", ".mlp1."), (r"\.fc2\.", ".mlp2."), (r"\.final_layer_norm\.", ".mlp_ln."),
    (r"^encoder\.layer_norm\.", "encoder.ln_post."), (r"^decoder\.layer_norm\.", "decoder.ln."),
    (r"^decoder\.embed_tokens\.", "decoder.token_embedding."), (r"^decoder\.embed_positions\.weight$", "decoder.positional_embedding"),
]


def mlx_key(key: str) -> str | None:
    """HF parameter name → mlx_whisper name; None for tensors mlx_whisper doesn't load."""
    if key.startswith("model.encoder.embed_positions") or key == "proj_out.weight":
        return None  # fixed sinusoids, recomputed / tied to the token embedding
    for a, b in RENAMES:
        key = re.sub(a, b, key)
    return key


def dims(config: dict) -> dict:
    return {"n_mels": config["num_mel_bins"], "n_audio_ctx": config["max_source_positions"], "n_audio_state": config["d_model"],
            "n_audio_head": config["encoder_attention_heads"], "n_audio_layer": config["encoder_layers"],
            "n_vocab": config["vocab_size"], "n_text_ctx": config["max_target_positions"], "n_text_state": config["d_model"],
            "n_text_head": config["decoder_attention_heads"], "n_text_layer": config["decoder_layers"]}


def main() -> None:
    import mlx.core as mx
    import numpy as np
    from safetensors import safe_open

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    weights = {}
    for f in sorted(src.glob("*.safetensors")):
        st = safe_open(str(f), framework="numpy")
        for k in st.keys():
            if (name := mlx_key(k)) is None:
                continue
            v = st.get_tensor(k).astype(np.float16)  # one tensor at a time: the fp32 shards are 6 GB
            if re.fullmatch(r"encoder\.conv\d\.weight", name):
                v = v.transpose(0, 2, 1)  # torch (out, in, kernel) → mlx (out, kernel, in)
            weights[name] = mx.array(v)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    (out / "config.json").write_text(json.dumps(dims(json.loads((src / "config.json").read_text()))))


if __name__ == "__main__":
    main()
