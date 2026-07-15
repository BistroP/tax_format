"""Steps 2-4: run the benchmark across models x conditions, score with the
three-bucket scorer, and write results.json (the artifact the explorer reads).

Run:  python -m eval.run_eval                 # MVP: prose + json, piloted items
      python -m eval.run_eval --all           # add lexical + strict_schema
      python -m eval.run_eval --no-pilot      # use the full stimulus set
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

from . import config, harness, scorer


def _load_items(use_pilot: bool):
    if use_pilot and config.PILOTED_PATH.exists():
        items = json.loads(config.PILOTED_PATH.read_text())["items"]
        print(f"Using piloted stimuli: {len(items)} items.")
        return items, True
    items = json.loads(config.STIMULI_PATH.read_text())["items"]
    if use_pilot:
        print("WARNING: no piloted_stimuli.json found — run `python -m eval.pilot` "
              "first. Falling back to the FULL, un-piloted set.")
    print(f"Using stimuli: {len(items)} items.")
    return items, False


def _rate(buckets, name):
    return round(sum(1 for b in buckets if b == name) / len(buckets), 4) if buckets else 0.0


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _metrics_for(rows):
    """rows: list of score_full dicts (each has bucket, pure_json, lexical_violation)."""
    buckets = [r["bucket"] for r in rows]
    pj = [1 if r["pure_json"] else 0 for r in rows if r["pure_json"] is not None]
    lv = [1 if r["lexical_violation"] else 0 for r in rows if r["lexical_violation"] is not None]
    return {
        "n": len(rows),
        "correct_rate": _rate(buckets, "correct"),
        "wrong_rate": _rate(buckets, "wrong"),
        "unparseable_rate": _rate(buckets, "unparseable"),
        "pure_json_compliance": _mean(pj),
        "lexical_violation_rate": _mean(lv),
    }


def run(conditions, use_pilot=True, use_cache=True):
    items, piloted = _load_items(use_pilot)
    if use_cache:
        print("Prefetching new calls in parallel...")
        harness.prefetch(config.MODELS, items, conditions)
    responses, scored = {}, {}  # scored[model][cond] -> list; scored_cat[model][cond][cat]

    total = len(config.MODELS) * len(conditions) * len(items)
    done = 0
    for m in config.MODELS:
        scored[m["key"]] = {c: [] for c in conditions}
        for cond in conditions:
            for item in items:
                rec = harness.query(m, item, cond, use_cache=use_cache)
                full = scorer.score_full(rec["text"], cond, item)
                key = f"{m['key']}|{item['id']}|{cond}"
                responses[key] = {
                    "text": rec["text"],
                    "bucket": full["bucket"],
                    "pure_json": full["pure_json"],
                    "lexical_violation": full["lexical_violation"],
                    "stop_reason": rec.get("stop_reason"),
                    "latency_s": rec.get("latency_s"),
                }
                scored[m["key"]][cond].append({**full, "category": item["category"]})
                done += 1
            print(f"  {m['key']:6} / {cond:12} done ({done}/{total})")

    # Aggregate metrics (overall + per category) and the tax.
    metrics, by_category, tax = {}, {}, {}
    categories = sorted({it["category"] for it in items})
    for m in config.MODELS:
        mk = m["key"]
        metrics[mk] = {c: _metrics_for(scored[mk][c]) for c in conditions}
        by_category[mk] = {
            c: {cat: _metrics_for([r for r in scored[mk][c] if r["category"] == cat])
                for cat in categories}
            for c in conditions
        }
        prose_correct = metrics[mk].get("prose", {}).get("correct_rate", 0.0)
        tax[mk] = {c: round(prose_correct - metrics[mk][c]["correct_rate"], 4)
                   for c in conditions}

    out = {
        "meta": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "models": [{"key": m["key"], "label": m["label"], "model": m["model"]}
                       for m in config.MODELS],
            "conditions": conditions,
            "categories": categories,
            "n_items": len(items),
            "piloted": piloted,
            "settings": {"max_tokens": config.MAX_TOKENS, "reasoning": config.REASONING,
                         "temperature": {m["key"]: m["temperature"] for m in config.MODELS}},
            "note": ("Format tax is an established effect (Tam 2024); this is an "
                     "independent, multi-vendor replication + instrument (via "
                     "OpenRouter). Instruct-only models so reasoning and formatting "
                     "compete for one budget (Fan 2026)."),
        },
        "items": [{"id": it["id"], "question": it["question"],
                   "canonical_answer": it["canonical_answer"],
                   "category": it["category"], "banned_word": it.get("banned_word", "")}
                  for it in items],
        "responses": responses,
        "metrics": metrics,
        "by_category": by_category,
        "tax": tax,
    }
    config.RESULTS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {config.RESULTS_PATH.relative_to(config.ROOT)}")
    _print_summary(out)
    return out


def _print_summary(out):
    print("\n=== The tax: correct_rate(prose) - correct_rate(condition) ===")
    conds = [c for c in out["meta"]["conditions"] if c != "prose"]
    header = f"{'model':16} {'prose✓':>7} " + " ".join(f"{c+'✓':>7} {c+'Δ':>6}" for c in conds)
    print(header)
    for m in out["meta"]["models"]:
        mk = m["key"]
        prose = out["metrics"][mk]["prose"]["correct_rate"]
        cells = ""
        for c in conds:
            cr = out["metrics"][mk][c]["correct_rate"]
            cells += f" {cr:7.2f} {out['tax'][mk][c]:+6.2f}"
        print(f"{m['label']:16} {prose:7.2f}{cells}")
    # Highlight the confound split for json.
    if "json" in out["meta"]["conditions"]:
        print("\nJSON compliance / unparseable (the confound we refuse to hide):")
        for m in out["meta"]["models"]:
            j = out["metrics"][m["key"]]["json"]
            pj = j["pure_json_compliance"]
            pj_s = f"{pj:.2f}" if pj is not None else "n/a"
            print(f"  {m['label']:16} unparseable={j['unparseable_rate']:.2f}  "
                  f"pure_json_compliance={pj_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include lexical + strict_schema")
    ap.add_argument("--no-pilot", action="store_true", help="use full un-piloted stimuli")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached responses")
    args = ap.parse_args()
    conditions = config.ALL_CONDITIONS if args.all else config.MVP_CONDITIONS
    run(conditions, use_pilot=not args.no_pilot, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
