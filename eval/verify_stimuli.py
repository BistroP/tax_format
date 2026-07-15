"""Independently re-verify the arithmetic answer keys: parse each question's TEXT
back into operations and recompute, then compare to canonical_answer. This is
independent of the generator that produced the items — if a key were wrong, this
would catch it. Word-problem / factual / two-hop items are hand-authored and not
machine-parseable; they are reported separately (and are pilot-gated in the run).

Run:  python -m eval.verify_stimuli      (exit code 1 if any mismatch)
"""
from __future__ import annotations

import json
import re
import sys

from . import config

_OP = re.compile(
    r"(add|subtract|increase by|decrease by|multiply by|multiply it by|"
    r"divide by|divide it by)\s+(\d+)", re.I)


def recompute_chain(q: str):
    """'Start with the number N. <op>, then <op>, ... What is the result?'"""
    m = re.match(r"start with the number (\d+)\.\s*(.+?)\.?\s*what is the result\?$",
                 q.strip(), re.I)
    if not m:
        return None
    v = int(m.group(1))
    for step in re.split(r",\s*then\s+", m.group(2)):
        mm = _OP.match(step.strip())
        if not mm:
            return ("UNPARSED_STEP", step.strip())
        op, k = mm.group(1).lower(), int(mm.group(2))
        if op in ("add", "increase by"):
            v += k
        elif op in ("subtract", "decrease by"):
            v -= k
        elif op in ("multiply by", "multiply it by"):
            v *= k
        else:  # divide
            if v % k:
                return ("NON_INTEGER_DIVISION", v, k)
            v //= k
    return v


def recompute_single(q: str):
    """'What is A times/plus/minus/divided by B?'"""
    m = re.match(r"what is (\d+) (times|plus|minus|divided by) (\d+)\?$",
                 q.strip(), re.I)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2).lower(), int(m.group(3))
    return {"times": a * b, "plus": a + b, "minus": a - b, "divided by": a // b}[op]


def main() -> int:
    items = json.loads(config.STIMULI_PATH.read_text())["items"]
    verified = mismatched = hand = 0
    problems = []
    for it in items:
        got = recompute_chain(it["question"])
        if got is None:
            got = recompute_single(it["question"])
        if got is None:
            hand += 1                       # word-problem / factual / two-hop
            continue
        if isinstance(got, tuple):
            problems.append((it["id"], "parser", got)); mismatched += 1
        elif str(got) == it["canonical_answer"]:
            verified += 1
        else:
            problems.append((it["id"], it["canonical_answer"], got)); mismatched += 1

    print(f"{verified} arithmetic items re-verified from their own question text · "
          f"{mismatched} mismatch(es) · {hand} hand-authored "
          f"(word-problem / factual / two-hop — pilot-gated, not machine-parseable)")
    for p in problems:
        print("  MISMATCH:", p)
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
