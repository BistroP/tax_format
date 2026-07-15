"""Top the ms_4step / ms_5step / ms_6step categories up to n=100 each.

Appends generated items to the existing ones rather than regenerating the set: the
current items keep their ids, so their cached responses stay valid and the top-up
only costs the new items. Every appended answer is computed by walking the op chain
(never hand-typed), and `python -m eval.verify_stimuli` re-derives it from the
question *text* afterwards as an independent check.

Note on composition: the existing 50/level are ~86% bare "Start with the number N…"
and ~14% narrative ("A theatre starts with 120 seats booked…"); the generator emits
only the bare form, so topping up to 100 shifts each level to ~93% bare. That's a
surface-form shift worth knowing about, but it can't touch the tax, which is a
*within-item* prose-vs-constraint difference — every condition sees the same items.

  python -m eval.make_multistep          # dry: report what would change
  python -m eval.make_multistep --write  # append to data/stimuli.json
"""

from __future__ import annotations

import json
import random
import sys

from . import config
from .ceiling_sweep import gen_one, question
from .numwords import variants

STIMULI = config.ROOT / "data" / "stimuli.json"
LEVELS = [4, 5, 6]
TARGET = 100
SEED = 20260715  # distinct from the sweep's; these levels don't overlap it anyway


def top_up(existing: list[dict], level: int) -> list[dict]:
    have = [it for it in existing if it["category"] == f"ms_{level}step"]
    seen = {it["question"] for it in have}
    rng = random.Random(SEED + level)
    out, i = [], len(have)
    while len(have) + len(out) < TARGET:
        start, ops, ans = gen_one(level, rng)
        q = question(start, ops)
        if q in seen:  # don't let the generator hand us the same chain twice
            continue
        seen.add(q)
        i += 1
        out.append({"id": f"ms{level}_{i:02d}", "question": q,
                    "canonical_answer": str(ans), "acceptable_variants": variants(ans),
                    "category": f"ms_{level}step", "banned_word": str(ans)})
    return out


def main(write: bool) -> None:
    doc = json.loads(STIMULI.read_text())
    items = doc["items"]
    ids = {it["id"] for it in items}

    added = []
    for level in LEVELS:
        new = top_up(items, level)
        clash = ids & {it["id"] for it in new}
        if clash:
            raise SystemExit(f"id collision at level {level}: {sorted(clash)[:5]}")
        have = sum(1 for it in items if it["category"] == f"ms_{level}step")
        print(f"ms_{level}step: {have} -> {have + len(new)}  (+{len(new)} generated)")
        if new:
            ex = new[0]
            print(f"    e.g. {ex['id']}: {ex['question'][:96]}…  -> {ex['canonical_answer']}")
        added += new

    print(f"\ntotal stimuli: {len(items)} -> {len(items) + len(added)}")
    if not write:
        print("\nDry run. Re-run with --write, then:")
        print("  python -m eval.verify_stimuli && python -m eval.pilot && python -m eval.run_eval --all")
        return

    doc["items"] = items + added
    STIMULI.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"\nWrote {STIMULI.relative_to(config.ROOT)} ({len(doc['items'])} items).")


if __name__ == "__main__":
    main("--write" in sys.argv)
