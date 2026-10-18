from yt_distill import asr, bakeoff, cli, distill, mlx_convert, youtube


def test_pack_merges_short_spans_and_splits_long_ones():
    chunks = asr.pack([(0, 3), (3.5, 6), (6.5, 30)], max_len=12)
    # 0–6 merges; the 23.5 s span splits in two ≤ 12 s pieces, the first of which joins nothing
    assert len(chunks) == 3
    assert all(b - a <= 12 + sum(asr.EDGE_PAD_S) for a, b in chunks)
    assert all(chunks[i][1] <= chunks[i + 1][0] for i in range(len(chunks) - 1)), "padding must not overlap"


def test_pack_pads_edges():
    (a, b), = asr.pack([(10, 15)])
    assert a == 10 - asr.EDGE_PAD_S[0] and b == 15 + asr.EDGE_PAD_S[1]


def test_quality_flags_loops():
    segs = [{"start": i, "end": i + 1, "text": "బట్ బట్"} for i in range(20)]
    assert asr.quality(segs, 600)["suspect"]


def test_vaani_runs_on_mlx_only_once_converted(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "MODELS", tmp_path)
    assert asr.mlx_dir("vaani", "te") is None  # → whisper.cpp fallback
    (tmp_path / "mlx-vaani-te").mkdir()
    (tmp_path / "mlx-vaani-te" / "weights.safetensors").write_bytes(b"x")
    assert asr.mlx_dir("vaani", "te") == tmp_path / "mlx-vaani-te"
    assert asr.mlx_dir("vaani", "hi") is None and asr.mlx_dir("turbo", "te") is None


def test_mlx_key_maps_hf_names():
    assert mlx_convert.mlx_key("model.decoder.layers.3.encoder_attn.k_proj.weight") == "decoder.blocks.3.cross_attn.key.weight"
    assert mlx_convert.mlx_key("model.encoder.layers.0.final_layer_norm.bias") == "encoder.blocks.0.mlp_ln.bias"
    assert mlx_convert.mlx_key("model.decoder.embed_positions.weight") == "decoder.positional_embedding"
    assert mlx_convert.mlx_key("model.encoder.layer_norm.weight") == "encoder.ln_post.weight"
    assert mlx_convert.mlx_key("proj_out.weight") is None and mlx_convert.mlx_key("model.encoder.embed_positions.weight") is None


def test_fmt_ts():
    assert asr.fmt_ts(75) == "01:15" and asr.fmt_ts(3725) == "1:02:05"


def test_cer_ignores_punctuation_and_spacing():
    assert bakeoff.cer("నార్మల్ క్యాండిల్, ఓకే!", "నార్మల్  క్యాండిల్ ఓకే") == 0
    assert 0 < bakeoff.cer("abcd", "abce") == 0.25


def test_fill_leaves_braces_in_values_alone():
    assert distill._fill("{a} {b} {c}", a="{b}", b="x") == "{b} x {c}"


def test_channel_url_points_at_uploads():
    assert youtube._normalize("https://www.youtube.com/@moneypurse") == "https://www.youtube.com/@moneypurse/videos"
    assert youtube._normalize("https://www.youtube.com/watch?v=x") == "https://www.youtube.com/watch?v=x"


def test_bare_url_defaults_to_distill(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.pipeline, "run", lambda *a: seen.setdefault("args", a) and 0)
    cli.main(["https://youtu.be/x", "--limit", "2"])
    assert seen["args"][0] == "https://youtu.be/x" and seen["args"][3] == 2 and seen["args"][1] is None


def test_score_skips_draft_references(tmp_path, monkeypatch):
    clip = tmp_path / "c1"
    clip.mkdir()
    (clip / "reference.txt").write_text(bakeoff.DRAFT + "\nనమస్కారం\n")
    (clip / "vaani.txt").write_text("నమస్కారం\n")
    monkeypatch.setattr(bakeoff, "BAKEOFF", tmp_path)
    bakeoff.score()
    assert not (tmp_path / "report.md").exists()
    (clip / "reference.txt").write_text("# status: corrected\nనమస్కారం\n")
    bakeoff.score()
    assert "0.0%" in (tmp_path / "report.md").read_text()


def test_model_routing_by_language():
    assert asr.default_model("te") == "vaani" and asr.default_model("hi") == "vaani"
    assert asr.default_model("en") == "turbo"
    assert asr.model_file("vaani", "hi").name == "ggml-vaani-large-v3-hi-q8_0.bin"
    assert asr.model_file("turbo", "auto").name == "ggml-large-v3-turbo.bin"
    assert asr.model_file("vaani", "ta") is None
    assert asr.model_file("indic", "te").name == "indic_worker.py" and asr.model_file("indic", "en") is None


def test_bakeoff_adds_auto_variant_when_turbo_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "MODELS", tmp_path)
    (tmp_path / "ggml-large-v3-turbo.bin").touch()
    assert bakeoff.candidates("hi") == [("turbo", "turbo", "hi"), ("indic", "indic", "hi"), ("turbo-auto", "turbo", "auto")]
    assert ("indic", "indic", "en") not in bakeoff.candidates("en")


def test_gpu_lock_serializes(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(asr, "MODELS", tmp_path)
    with asr.gpu_lock():
        with open(tmp_path / ".asr.lock") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                raise AssertionError("lock should be held")
            except BlockingIOError:
                pass


def test_mixed_script_keeps_original_batch_if_lines_are_lost(monkeypatch):
    monkeypatch.setattr(distill, "MIXED_BATCH_LINES", 2)
    replies = iter(["[00:01] a in Latin\n[00:02] b in Latin", "[00:03] only one line back"])
    monkeypatch.setattr(distill, "claude", lambda prompt: next(replies))
    text = "[00:01] a\n[00:02] b\n[00:03] c\n[00:04] d"
    assert distill.mixed_script(text, "te").splitlines() == ["[00:01] a in Latin", "[00:02] b in Latin", "[00:03] c", "[00:04] d"]
