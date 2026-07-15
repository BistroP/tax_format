"""Reasoning-ceiling sweep: push the arithmetic ladder WAY past 6 steps and, UNGATED,
measure accuracy vs depth for prose / json / strict_schema. The point is to find
where *prose* (chain-of-thought allowed) itself breaks — the reasoning ceiling — and
contrast it with the answer-only-json floor.

CRITICAL: uses a big max_tokens (4096) so long chains-of-thought aren't truncated —
otherwise the "ceiling" is just a token limit, not a reasoning limit.

  python -m eval.ceiling_sweep          # dry: generate + validate, no API
  python -m eval.ceiling_sweep --run     # full sweep (writes ceiling.json + ceiling.html)
"""
from __future__ import annotations

import json
import random
import sys

from . import config, scorer

LEVELS = [8, 12, 16, 20, 28, 40]
PER_LEVEL = 12
CONDITIONS = ["prose", "json", "strict_schema"]
MAX_TOKENS = 4096
SEED = 20260708

ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
        "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
def _two(n):
    if n < 20: return [ONES[n]]
    t = TENS[n//10]
    return [t] if n % 10 == 0 else [f"{t}-{ONES[n%10]}", f"{t} {ONES[n%10]}"]
def _three(n):
    if n < 100: return _two(n)
    h = ONES[n//100] + " hundred"; rem = n % 100
    if rem == 0: return [h]
    out = []
    for r in _two(rem): out += [f"{h} {r}", f"{h} and {r}"]
    return out
def variants(n): return sorted(set(_three(n)))

PHRASE = {"add": lambda k: f"add {k}", "sub": lambda k: f"subtract {k}",
          "mul": lambda k: f"multiply by {k}", "div": lambda k: f"divide by {k}"}

def gen_one(level, rng):
    for _ in range(20000):
        v = rng.randint(20, 150); start = v; ops = []; ok = True
        for _ in range(level):
            ch = [("add", rng.randint(10, 80))]
            if v > 25: ch.append(("sub", rng.randint(10, v - 15)))
            for m in (2, 3):
                if v * m <= 999: ch.append(("mul", m))
            for d in (2, 3, 4, 5):
                if v % d == 0 and v // d >= 10: ch.append(("div", d))
            o, k = rng.choice(ch); ops.append((o, k))
            v = v + k if o == "add" else v - k if o == "sub" else v * k if o == "mul" else v // k
            if not (10 <= v <= 999): ok = False; break
        if ok and len(ops) == level:
            return start, ops, v
    raise RuntimeError(f"gen failed at level {level}")

def question(start, ops):
    body = ", then ".join(PHRASE[o](k) for o, k in ops)
    return f"Start with the number {start}. {body[0].upper() + body[1:]}. What is the result?"

def build_items():
    rng = random.Random(SEED)
    items = []
    for level in LEVELS:
        for i in range(PER_LEVEL):
            start, ops, ans = gen_one(level, rng)
            items.append({"id": f"deep{level:02d}_{i:02d}", "question": question(start, ops),
                          "canonical_answer": str(ans), "acceptable_variants": variants(ans),
                          "category": f"ms_{level}step", "banned_word": str(ans), "_steps": level})
    return items

def _pct(x): return f"{100*x:5.0f}%"

def run():
    from . import harness
    items = build_items()
    print(f"Sweep: {len(items)} items · levels {LEVELS} · {len(config.MODELS)} models · "
          f"conditions {CONDITIONS} · max_tokens {MAX_TOKENS}")
    harness.prefetch(config.MODELS, items, CONDITIONS, max_tokens=MAX_TOKENS)

    # acc[cond][level] = list of 0/1 across (model,item); accm[cond][level][model_key] too
    acc = {c: {L: [] for L in LEVELS} for c in CONDITIONS}
    accm = {c: {L: {m["key"]: [] for m in config.MODELS} for L in LEVELS} for c in CONDITIONS}
    for it in items:
        L = it["_steps"]
        for m in config.MODELS:
            for c in CONDITIONS:
                rec = harness.query(m, it, c, max_tokens=MAX_TOKENS)
                hit = 1 if scorer.score(rec["text"], c, it) == "correct" else 0
                acc[c][L].append(hit); accm[c][L][m["key"]].append(hit)

    def rate(lst): return sum(lst) / len(lst) if lst else 0.0
    curve = {c: {L: rate(acc[c][L]) for L in LEVELS} for c in CONDITIONS}

    print(f"\n=== Accuracy by depth (UNGATED · n={PER_LEVEL}/level · avg over {len(config.MODELS)} models) ===")
    print("steps  " + "  ".join(f"{c[:6]:>7}" for c in CONDITIONS))
    for L in LEVELS:
        print(f"{L:4}   " + "  ".join(f"{_pct(curve[c][L])}" for c in CONDITIONS))

    print("\n=== PROSE accuracy per model (where does each model's reasoning break?) ===")
    print("steps  " + "  ".join(f"{m['key'][:8]:>8}" for m in config.MODELS))
    for L in LEVELS:
        print(f"{L:4}   " + "  ".join(f"{_pct(rate(accm['prose'][L][m['key']])):>8}" for m in config.MODELS))

    out = {"levels": LEVELS, "per_level": PER_LEVEL, "conditions": CONDITIONS,
           "models": [m["key"] for m in config.MODELS], "curve": curve,
           "prose_by_model": {L: {m["key"]: rate(accm["prose"][L][m["key"]]) for m in config.MODELS} for L in LEVELS}}
    (config.ROOT / "ceiling.json").write_text(json.dumps(out, indent=2))
    _write_chart(curve)
    print(f"\nWrote ceiling.json + ceiling.html")

def _write_chart(curve):
    W, H, pad = 720, 420, 60
    xs = LEVELS
    def X(s): return pad + (s - xs[0]) / (xs[-1] - xs[0]) * (W - 2 * pad)
    def Y(a): return H - pad - a * (H - 2 * pad)
    colors = {"prose": "#157f4b", "json": "#c0392b", "strict_schema": "#2d6cdf"}
    lines = ""
    for c in CONDITIONS:
        pts = " ".join(f"{X(L):.0f},{Y(curve[c][L]):.0f}" for L in xs)
        col = colors.get(c, "#666")
        lines += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3"/>'
        for L in xs:
            lines += f'<circle cx="{X(L):.0f}" cy="{Y(curve[c][L]):.0f}" r="4" fill="{col}"/>'
    yaxis = ""
    for a in (0, 0.25, 0.5, 0.75, 1.0):
        yaxis += (f'<line x1="{pad}" y1="{Y(a):.0f}" x2="{W-pad}" y2="{Y(a):.0f}" stroke="#eee"/>'
                  f'<text x="{pad-8}" y="{Y(a)+4:.0f}" text-anchor="end" font-size="12" fill="#888">{int(a*100)}%</text>')
    xaxis = "".join(f'<text x="{X(L):.0f}" y="{H-pad+20:.0f}" text-anchor="middle" font-size="12" fill="#888">{L}</text>' for L in xs)
    legend = ""
    for i, c in enumerate(CONDITIONS):
        legend += (f'<rect x="{W-pad-150}" y="{pad+i*20}" width="12" height="12" fill="{colors.get(c,"#666")}"/>'
                   f'<text x="{W-pad-132}" y="{pad+i*20+11}" font-size="13" fill="#333">{c}</text>')
    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui">'
           f'<rect width="{W}" height="{H}" fill="#fff"/>{yaxis}{lines}{xaxis}{legend}'
           f'<text x="{W/2:.0f}" y="{H-12}" text-anchor="middle" font-size="13" fill="#555">reasoning steps</text>'
           f'<text x="{W/2:.0f}" y="24" text-anchor="middle" font-size="15" font-weight="700">The reasoning ceiling: accuracy vs depth (avg over 5 models)</text></svg>')
    (config.ROOT / "ceiling.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>Reasoning ceiling</title>{svg}")

def dry():
    rng = random.Random(SEED)
    print("Dry run — generating + validating (no API):")
    for level in LEVELS:
        start, ops, ans = gen_one(level, rng)
        # independent recompute
        v = start
        for o, k in ops:
            v = v + k if o == "add" else v - k if o == "sub" else v * k if o == "mul" else v // k
        assert v == ans, (level, v, ans)
        print(f"  {level:2}-step -> {ans:>3}  | {question(start, ops)[:100]}...")
    print(f"OK. {len(LEVELS)} levels x {PER_LEVEL} = {len(LEVELS)*PER_LEVEL} items would be generated.")

if __name__ == "__main__":
    (run if "--run" in sys.argv else dry)()
