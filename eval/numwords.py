"""Spell integers 0-999 as words, every reasonable way.

These become an item's `acceptable_variants`, which matters twice:
  - `lexical` bans the digit form, so "fifty-six" is the *intended* compliant answer
    and has to be creditable;
  - a prose model may write either form unprompted.
Both spellings of the hyphen case ("fifty-six" / "fifty six") and both of the
hundreds case ("two hundred eight" / "two hundred and eight") are accepted, since
which one a model reaches for is a coin flip we shouldn't be scoring.
"""

from __future__ import annotations

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _two(n: int) -> list[str]:
    if n < 20:
        return [ONES[n]]
    t = TENS[n // 10]
    return [t] if n % 10 == 0 else [f"{t}-{ONES[n % 10]}", f"{t} {ONES[n % 10]}"]


def _three(n: int) -> list[str]:
    if n < 100:
        return _two(n)
    h = ONES[n // 100] + " hundred"
    rem = n % 100
    if rem == 0:
        return [h]
    out = []
    for r in _two(rem):
        out += [f"{h} {r}", f"{h} and {r}"]
    return out


def variants(n: int) -> list[str]:
    """All word spellings of n (0-999), deduped and sorted."""
    return sorted(set(_three(n)))
