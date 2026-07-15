"""Step 1 (the non-optional gate): pilot every stimulus in the PROSE condition and
keep only the items ALL target models get right unconstrained. No prose baseline
=> no measurable drop => the whole experiment is meaningless (Section 8).

Run:  python -m eval.pilot
Writes data/piloted_stimuli.json (the kept subset) + prints a report.
"""

from __future__ import annotations

import json

from . import config, harness, scorer


def run_pilot(use_cache: bool = True) -> dict:
    stimuli = json.loads(config.STIMULI_PATH.read_text())["items"]
    kept, dropped, per_item = [], [], {}

    if use_cache:
        harness.prefetch(config.MODELS, stimuli, ["prose"])
    print(f"Piloting {len(stimuli)} items in PROSE across "
          f"{len(config.MODELS)} models...\n")
    for item in stimuli:
        row = {}
        all_correct = True
        for m in config.MODELS:
            rec = harness.query(m, item, "prose", use_cache=use_cache)
            bucket = scorer.score(rec["text"], "prose", item)
            row[m["key"]] = bucket
            all_correct = all_correct and (bucket == "correct")
        per_item[item["id"]] = row
        (kept if all_correct else dropped).append(item)
        flag = "keep" if all_correct else "DROP"
        marks = " ".join(f"{k}={v[:4]}" for k, v in row.items())
        print(f"  {flag}  {item['id']:8} {marks}")

    config.PILOTED_PATH.write_text(
        json.dumps({"items": kept}, indent=2, ensure_ascii=False)
    )
    print(f"\nKept {len(kept)}/{len(stimuli)} items "
          f"(dropped {len(dropped)} that a target model missed in prose).")
    print(f"Wrote {config.PILOTED_PATH.relative_to(config.ROOT)}")
    if dropped:
        print("Dropped:", ", ".join(i["id"] for i in dropped))
    return {"kept": kept, "dropped": dropped, "per_item": per_item}


if __name__ == "__main__":
    run_pilot()
