"""Print OpenRouter's live model catalog so you can pick or verify slugs before
putting them in config.MODELS. The models endpoint is public — no key needed.

Usage:  python -m eval.list_models              # all models
        python -m eval.list_models llama        # filter by substring
        python -m eval.list_models :free         # only free-tier variants
"""

from __future__ import annotations

import json
import sys
import urllib.request

URL = "https://openrouter.ai/api/v1/models"


def main() -> None:
    needle = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    with urllib.request.urlopen(URL, timeout=30) as r:
        data = json.load(r)["data"]

    rows = []
    for m in data:
        mid = m["id"]
        if needle and needle not in mid.lower():
            continue
        pricing = m.get("pricing", {}) or {}
        p_in = str(pricing.get("prompt", "?"))
        p_out = str(pricing.get("completion", "?"))
        free = ":free" in mid or p_in in ("0", "0.0")
        rows.append((mid, "FREE" if free else f"in={p_in} out={p_out} $/tok"))

    for mid, price in sorted(rows):
        print(f"  {mid:52} {price}")
    tail = f" matching '{needle}'" if needle else ""
    print(f"\n{len(rows)} model(s){tail} of {len(data)} total.")


if __name__ == "__main__":
    main()
