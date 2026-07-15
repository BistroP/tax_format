"""Build the GSM8K slice of data/stimuli.json from the OFFICIAL test split.

Why fetch rather than transcribe: the whole point of the GSM8K slice is that the
labels are *not ours* — it's the check that the effect isn't an artifact of chains
we wrote. Hand-copying problems puts our typing back between the dataset and the
result, which is exactly what the slice exists to rule out. So we pull test.jsonl
and parse the `#### N` gold labels programmatically. Item ids carry the source line
(gsm_0123 = line 123 of test.jsonl), so any item is traceable back to the dataset.

Two declared filters (both in the METHODOLOGY, neither silent):
  - answers restricted to 0-999, which preserves the answer-length control the rest
    of the design rests on (1-3 chars, as everywhere else). Costs ~10% of the pool
    and, incidentally, removes every comma-formatted label ("1,000") — which the
    scorer would otherwise fail to match against a canonical "1000".
  - the usual pilot gate runs afterwards, as for every other category.

  python -m eval.make_gsm8k          # dry: report what would change
  python -m eval.make_gsm8k --write  # rewrite the gsm8k items in data/stimuli.json
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.request

from . import config
from .numwords import variants

SRC = ("https://raw.githubusercontent.com/openai/grade-school-math/"
       "master/grade_school_math/data/test.jsonl")
LOCAL = config.ROOT / "data" / "gsm8k_test.jsonl"
STIMULI = config.ROOT / "data" / "stimuli.json"
N = 100
SEED = 20260715
MAX_ANSWER = 999  # keep answer length at 1-3 chars, as in every other category

LABEL = re.compile(r"####\s*(-?[\d,]+)\s*$")


def fetch() -> list[dict]:
    if not LOCAL.exists():
        print(f"Fetching {SRC}")
        urllib.request.urlretrieve(SRC, LOCAL)
    rows = [json.loads(l) for l in LOCAL.read_text().splitlines() if l.strip()]
    print(f"GSM8K test split: {len(rows)} problems ({LOCAL.name})")
    return rows


def candidates(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        m = LABEL.search(r["answer"])
        if not m:
            continue
        n = int(m.group(1).replace(",", ""))
        if not (0 <= n <= MAX_ANSWER):
            continue
        out.append({
            "id": f"gsm_{i:04d}",
            "question": r["question"].strip(),
            "canonical_answer": str(n),
            "acceptable_variants": variants(n),
            "category": "gsm8k",
            "banned_word": str(n),
        })
    return out


def build() -> list[dict]:
    pool = candidates(fetch())
    print(f"  {len(pool)} have an integer gold label in [0, {MAX_ANSWER}] "
          f"-> sampling {N} (seed {SEED})")
    return sorted(random.Random(SEED).sample(pool, N), key=lambda it: it["id"])


def main(write: bool) -> None:
    items = build()
    doc = json.loads(STIMULI.read_text())
    old = [it for it in doc["items"] if it["category"] == "gsm8k"]
    keep = [it for it in doc["items"] if it["category"] != "gsm8k"]

    print(f"\ngsm8k items: {len(old)} (hand-transcribed) -> {len(items)} (fetched)")
    print(f"total stimuli: {len(doc['items'])} -> {len(keep) + len(items)}")
    for it in items[:3]:
        q = it["question"].replace("\n", " ")
        print(f"\n  {it['id']} -> {it['canonical_answer']}  (variants: {it['acceptable_variants']})")
        print(f"    {q[:150]}{'…' if len(q) > 150 else ''}")

    if not write:
        print("\nDry run. Re-run with --write to update data/stimuli.json,")
        print("then: python -m eval.pilot && python -m eval.run_eval --all")
        return

    doc["items"] = keep + items
    STIMULI.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"\nWrote {STIMULI.relative_to(config.ROOT)} ({len(doc['items'])} items).")
    print("Next: python -m eval.pilot && python -m eval.run_eval --all")


if __name__ == "__main__":
    main("--write" in sys.argv)
