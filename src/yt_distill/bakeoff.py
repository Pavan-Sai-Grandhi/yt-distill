"""Compare ASR models on short clips against references you correct by hand.

prepare: cut a clip from each video, transcribe it with every installed model, and seed
         bakeoff/<clip>/reference.txt from the default model's output for you to correct.
score:   character error rate (CER) of each model against the corrected references.
sheet:   one phone-friendly HTML page with each clip's outputs side by side, blind-labelled
         (key in bakeoff/blind_key.json), for a quick "which reads closest" judgement.
"""

import html
import json
import random
import re
import time
import unicodedata
from pathlib import Path

from . import asr, youtube
from .paths import ROOT

BAKEOFF = ROOT / "bakeoff"
DRAFT = "# status: draft — fix this text to match exactly what is said, then change 'draft' to 'corrected'"


def candidates(language: str) -> list[tuple[str, str, str]]:
    """(label, model, language) for every installed model that can do this language."""
    out = [(m, m, language) for m in asr.ASR_MODELS if (f := asr.model_file(m, language)) and f.exists()]
    if any(m == "turbo" for _, m, _ in out):
        out.append(("turbo-auto", "turbo", "auto"))  # per-chunk language ID; suits code-mixed speech
    return out


def prepare(urls: list[str], start: float | None, length: float, language: str) -> None:
    models = candidates(language)
    for url in urls:
        try:
            vid = youtube.resolve(url).videos[0]["id"]
            meta = youtube.fetch_audio(vid)
        except Exception as e:  # one unavailable video shouldn't sink the rest
            print(f"✗ {url}: {e}")
            continue
        at = start if start is not None else round((meta.get("duration") or 600) * 0.4)
        clip = BAKEOFF / f"{vid}_{int(at)}"
        clip.mkdir(parents=True, exist_ok=True)
        wav = clip / "clip.wav"
        if not wav.exists():  # write-then-rename: an interrupted cut must not look finished
            asr.to_wav(BAKEOFF.parent / "data" / vid / "audio.opus", clip / "clip.part.wav", at, length)
            (clip / "clip.part.wav").rename(wav)
        (clip / "source.json").write_text(json.dumps({"id": vid, "title": meta["title"], "url": f"{meta['url']}&t={int(at)}", "start": at, "length": length, "language": language}, ensure_ascii=False, indent=1))
        print(f"{clip.name}: {meta['title']}")
        for label, m, lang in models:
            out = clip / f"{label}.txt"
            if out.exists():
                continue
            t0 = time.monotonic()
            text = "\n".join(s["text"] for s in asr.transcribe(wav, m, lang))
            secs = time.monotonic() - t0
            out.write_text(text + "\n")
            timing = clip / "timing.json"
            t = json.loads(timing.read_text()) if timing.exists() else {}
            t[label] = round(length / secs, 1)
            timing.write_text(json.dumps(t))
            print(f"  {label}: {t[label]}x realtime")
        ref = clip / "reference.txt"
        if not ref.exists():
            seed = clip / f"{asr.default_model(language)}.txt"
            ref.write_text(DRAFT + "\n" + (seed.read_text() if seed.exists() else ""))
    print(f"\nListen to each bakeoff/<clip>/clip.wav (or the source.json URL), correct reference.txt, then run: yt-distill bakeoff score")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", " ", text)  # \w keeps Telugu letters and combining marks
    return " ".join(text.lower().split())


def cer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(len(r), 1)


def score() -> None:
    rows, models = [], set()
    for clip in sorted(p for p in BAKEOFF.glob("*") if (p / "reference.txt").exists()):
        lines = (clip / "reference.txt").read_text().splitlines()
        if not lines or not lines[0].startswith("# status: corrected"):
            print(f"skip {clip.name} (reference still draft)")
            continue
        ref = "\n".join(lines[1:])
        timing = json.loads((clip / "timing.json").read_text()) if (clip / "timing.json").exists() else {}
        res = {m.stem: cer(ref, m.read_text()) for m in clip.glob("*.txt") if m.stem != "reference"}
        models |= res.keys()
        rows.append((clip.name, res, timing))
    if not rows:
        print("No corrected references yet.")
        return
    models = sorted(models)
    lines = ["| clip | " + " | ".join(f"{m} CER" for m in models) + " |", "|---" * (len(models) + 1) + "|"]
    for name, res, _ in rows:
        lines.append(f"| {name} | " + " | ".join(f"{res[m]:.1%}" if m in res else "–" for m in models) + " |")
    mean = {m: sum(r[m] for _, r, _ in rows if m in r) / max(sum(m in r for _, r, _ in rows), 1) for m in models}
    lines.append("| **mean** | " + " | ".join(f"**{mean[m]:.1%}**" for m in models) + " |")
    rtf = {m: [t[m] for *_, t in rows if m in t] for m in models}
    lines.append("| speed (×realtime) | " + " | ".join(f"{sum(v) / len(v):.1f}" if v else "–" for v in rtf.values()) + " |")
    report = "\n".join(lines)
    (BAKEOFF / "report.md").write_text("# ASR bake-off\n\nLower CER is better.\n\n" + report + "\n")
    print(report)


def sheet(models: list[str] | None = None) -> None:
    clips = sorted(p for p in BAKEOFF.glob("*") if (p / "source.json").exists() and any(p.glob("*.txt")))
    key, cards = {}, []
    for n, clip in enumerate(clips, 1):
        src = json.loads((clip / "source.json").read_text())
        outs = sorted(p for p in clip.glob("*.txt") if p.stem != "reference" and (not models or p.stem in models))
        random.Random(clip.name).shuffle(outs)  # stable per clip, unrelated to model order
        key[clip.name] = {chr(65 + i): p.stem for i, p in enumerate(outs)}
        blocks = "".join(
            f'<div class="out" data-label="{chr(65 + i)}"><label class="pick"><input type="radio" name="best-{clip.name}" value="{chr(65 + i)}"> '
            f'<b>Output {chr(65 + i)}</b> is closest</label><span class="count"></span><p>'
            + " ".join(f"<span class=w>{html.escape(w)}</span>" for w in p.read_text().split())
            + "</p></div>"
            for i, p in enumerate(outs)
        )
        cards.append(
            f'<section class="clip" data-clip="{clip.name}"><h2>Clip {n} · {src.get("language", "te")}</h2>'
            f'<p class="t">{html.escape(src["title"])} · <a href="{html.escape(src["url"])}">YouTube</a></p>'
            f'<audio controls preload="none" src="{clip.name}/clip.wav"></audio>'
            f'<div class="grid">{blocks}</div><textarea placeholder="Optional note on this clip"></textarea></section>'
        )
    (BAKEOFF / "blind_key.json").write_text(json.dumps(key, indent=1))
    (BAKEOFF / "sheet.html").write_text(_SHEET.replace("{{CARDS}}", "".join(cards)))
    print(f"{BAKEOFF / 'sheet.html'} ({len(clips)} clips)")


_SHEET = r"""<!doctype html><html lang="en" data-lavish-annotate="off"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ASR bake-off sheet</title><style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f5f5f7;--line:#d2d2d7;--acc:#0a66c2;--bad:#ffd6d6;--badfg:#9b1c1c;--sel:#0a66c2}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#f2f2f2;--mut:#a1a1a6;--card:#1c1c1e;--line:#3a3a3c;--acc:#5aa9ff;--bad:#5c1f1f;--badfg:#ffb4b4;--sel:#5aa9ff}}
*{box-sizing:border-box} body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:auto} h1{font-size:1.4rem;margin:0 0 4px} .mut,.t{color:var(--mut)} a{color:var(--acc)}
section{border-top:1px solid var(--line);padding:16px 0} h2{margin:0;font-size:1.1rem} audio{width:100%;margin:8px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.out{background:var(--card);border-radius:10px;padding:12px;min-width:0;border:2px solid transparent}
.out.best{border-color:var(--sel)} .out p{margin:8px 0 0;overflow-wrap:anywhere}
.pick{cursor:pointer} .count{float:right;color:var(--mut);font-size:.85rem}
.w{cursor:pointer;border-radius:3px;padding:0 1px} .w:hover{outline:1px dashed var(--mut)}
.w.bad{background:var(--bad);color:var(--badfg);text-decoration:line-through}
textarea{width:100%;margin-top:10px;min-height:2.6em;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit}
.bar{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 0;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff} #out{width:100%;display:none}
</style></head><body><main>
<h1>ASR bake-off: which output is closest?</h1>
<p class="mut">For each clip: play the audio, <b>pick the closest output</b>, and <b>click any wrong word</b> to mark it (click again to unmark).
Labels are shuffled per clip. Your marks are saved in this browser, so you can stop and come back.</p>
{{CARDS}}
<div class="bar"><button class="primary" id="send">Send feedback to Claude</button><button id="copy">Copy as text</button>
<span id="status" class="mut" role="status"></span><textarea id="out" readonly></textarea></div>
</main><script>
const KEY = "ytd-bakeoff";
const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } };
const save = (st) => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch {} };
let st = load();
const clips = [...document.querySelectorAll(".clip")];

function counts(out) {
  const ws = out.querySelectorAll(".w"), bad = out.querySelectorAll(".w.bad").length;
  out.querySelector(".count").textContent = `${bad} wrong / ${ws.length} words`;
}
function persist(clip) {
  const id = clip.dataset.clip, outs = [...clip.querySelectorAll(".out")];
  st[id] = {
    best: clip.querySelector("input:checked")?.value || null,
    note: clip.querySelector("textarea").value,
    wrong: Object.fromEntries(outs.map(o => [o.dataset.label, [...o.querySelectorAll(".w")].flatMap((w, i) => w.classList.contains("bad") ? [i] : [])])),
  };
  save(st);
}
for (const clip of clips) {
  const saved = st[clip.dataset.clip] || {};
  for (const out of clip.querySelectorAll(".out")) {
    const idx = new Set((saved.wrong || {})[out.dataset.label] || []);
    out.querySelectorAll(".w").forEach((w, i) => {
      if (idx.has(i)) w.classList.add("bad");
      w.addEventListener("click", () => { w.classList.toggle("bad"); counts(out); persist(clip); });
    });
    const r = out.querySelector("input");
    if (saved.best === out.dataset.label) { r.checked = true; out.classList.add("best"); }
    r.addEventListener("change", () => {
      clip.querySelectorAll(".out").forEach(o => o.classList.toggle("best", o === out)); persist(clip);
    });
    counts(out);
  }
  const ta = clip.querySelector("textarea");
  ta.value = saved.note || "";
  ta.addEventListener("input", () => persist(clip));
}
function summary() {
  const items = [], lines = [];
  clips.forEach((clip, n) => {
    const id = clip.dataset.clip, s = st[id] || {};
    const outs = [...clip.querySelectorAll(".out")].map(o => ({
      label: o.dataset.label, words: o.querySelectorAll(".w").length,
      wrong: [...o.querySelectorAll(".w.bad")].map(w => w.textContent),
    }));
    if (!s.best && !outs.some(o => o.wrong.length) && !s.note) return;
    items.push({ clip: id, best: s.best, note: s.note || "", outputs: outs });
    lines.push(`Clip ${n + 1} (${id}): closest=${s.best || "?"}; ` +
      outs.map(o => `${o.label} ${o.wrong.length}/${o.words} wrong` + (o.wrong.length ? ` [${o.wrong.join(", ")}]` : "")).join("; ") +
      (s.note ? `; note: ${s.note}` : ""));
  });
  return { items, text: lines.join("\n") };
}
const status = (m) => document.getElementById("status").textContent = m;
document.getElementById("send").addEventListener("click", () => {
  const { items, text } = summary();
  if (!items.length) return status("Nothing judged yet.");
  if (window.lavish?.queuePrompt) {
    window.lavish.queuePrompt("Bake-off judgements (blind labels; map with bakeoff/blind_key.json), record and score them:\n" + text,
      { tag: "bakeoff", text: `Judged ${items.length} clip(s)`, data: { items } });
    window.lavish.sendQueuedPrompts?.();
    status(`Sent ${items.length} clip(s).`);
  } else {
    const o = document.getElementById("out"); o.value = text; o.style.display = "block"; o.select();
    status("Not opened through Lavish: copy this text and paste it in the chat.");
  }
});
document.getElementById("copy").addEventListener("click", async () => {
  const { items, text } = summary();
  if (!items.length) return status("Nothing judged yet.");
  try { await navigator.clipboard.writeText(text); status("Copied. Paste it in the chat."); }
  catch { const o = document.getElementById("out"); o.value = text; o.style.display = "block"; o.select(); status("Copy the text below."); }
});
</script></body></html>
"""
