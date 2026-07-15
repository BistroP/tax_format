"""The three-bucket scorer (Section 5.3).

For each response we produce exactly one of: ``correct`` | ``wrong`` | ``unparseable``.
``unparseable`` is reported SEPARATELY and is never folded into ``wrong`` — that
separation is the load-bearing rigor of the project: it tells "the model reasoned
wrong" apart from "I couldn't parse the model's output".

We also track, independently of correctness:
  * pure_json_compliance  — did the ENTIRE response parse as JSON, no surrounding prose?
  * lexical_violation     — (lexical condition) did the response use the banned word?
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
_EDGE_PUNCT = " \t\r\n\"'`.,:;!?()[]{}<>*_-"


def normalize(s: Any) -> str:
    """Lowercase, strip accents, strip surrounding whitespace/punctuation,
    collapse internal whitespace. Accent-folding lets 'Brasília' match 'Brasilia'."""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(_EDGE_PUNCT)
    return s


def accepted_set(item: dict) -> set[str]:
    values = [item["canonical_answer"], *item.get("acceptable_variants", [])]
    return {normalize(v) for v in values if str(v).strip()}


def _word_in(needle: str, haystack: str) -> bool:
    """Whole-token match: '6' must not match inside '60', nor 'six' inside 'sixty'.
    Lookarounds (not \\b) so multi-word needles like 'mexico city' still work."""
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


# --------------------------------------------------------------------------- #
# Lenient JSON extraction
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_code_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block (respecting strings), or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _loads_lenient(block: str) -> Optional[Any]:
    try:
        return json.loads(block)
    except Exception:
        pass
    try:
        return json.loads(_TRAILING_COMMA_RE.sub(r"\1", block))
    except Exception:
        return None


def try_extract_json(text: str) -> Optional[Any]:
    """Lenient: strip markdown fences, find first {...} block, tolerate trailing
    commas. Returns the parsed object or None. Used for correctness bucketing."""
    for candidate in (_strip_code_fences(text), text):
        block = _first_json_object(candidate)
        if block is not None:
            parsed = _loads_lenient(block)
            if parsed is not None:
                return parsed
    return None


def _extract_answer_field(parsed: Any) -> Optional[Any]:
    """Find the 'answer' field at top level, or nested one level deep."""
    if isinstance(parsed, dict):
        if "answer" in parsed:
            return parsed["answer"]
        for v in parsed.values():
            if isinstance(v, dict) and "answer" in v:
                return v["answer"]
    return None


def extract_answer(text: str) -> Optional[str]:
    """Best-effort: pull the 'answer' field out of a JSON-ish response, for
    display in the live panel (independent of correctness). None if not found."""
    ans = _extract_answer_field(try_extract_json(text))
    return None if ans is None else str(ans)


def is_pure_json(text: str) -> bool:
    """Did the WHOLE response parse as JSON with no surrounding prose or fences?
    Distinct from 'could I fish valid JSON out of it'."""
    try:
        obj = json.loads(text.strip())
    except Exception:
        return False
    return isinstance(obj, (dict, list))


def contains_banned_word(text: str, banned_word: str) -> bool:
    if not banned_word:
        return False
    return normalize(banned_word) in normalize(text)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
JSON_CONDITIONS = {"json", "strict_schema"}


def score(response_text: str, condition: str, item: dict) -> str:
    """Return exactly one of 'correct' | 'wrong' | 'unparseable'."""
    accepted = accepted_set(item)

    if condition in JSON_CONDITIONS:
        parsed = try_extract_json(response_text)
        answer = _extract_answer_field(parsed)
        if answer is None:
            return "unparseable"
        # Whole-token match INSIDE the answer field, so "$18" / "3 bolts" / "60%"
        # credit the value 18 / 3 / 60 (GSM8K answers carry units), while "108" in
        # a wrong "104" still fails. Boundary-safe: "18" won't match "180".
        candidate = normalize(str(answer))
        return "correct" if any(a and _word_in(a, candidate) for a in accepted) else "wrong"

    # prose / lexical — free-form; whole-token match (Section 5.3, boundary-safe)
    norm = normalize(response_text)
    return "correct" if any(a and _word_in(a, norm) for a in accepted) else "wrong"


def score_full(response_text: str, condition: str, item: dict) -> dict:
    """Bucket plus the separately-reported compliance signals."""
    result = {
        "bucket": score(response_text, condition, item),
        "pure_json": None,
        "lexical_violation": None,
    }
    if condition in JSON_CONDITIONS:
        result["pure_json"] = is_pure_json(response_text)
    if condition == "lexical":
        result["lexical_violation"] = contains_banned_word(
            response_text, item.get("banned_word", "")
        )
    return result


if __name__ == "__main__":  # tiny smoke test
    demo = {"canonical_answer": "Canberra", "acceptable_variants": [], "banned_word": "Canberra"}
    print(score("The capital is Canberra.", "prose", demo))          # correct
    print(score('```json\n{"answer": "Canberra"}\n```', "json", demo))  # correct
    print(score("I think it is Sydney.", "prose", demo))             # wrong
    print(score("Here you go, hope it helps!", "json", demo))        # unparseable
