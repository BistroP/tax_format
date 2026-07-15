"""Offline unit tests for the three-bucket scorer. No API keys needed.

Run: `python tests/test_scorer.py`  (or `pytest tests/`)
The central invariant under test: a JSON parse failure is `unparseable`, NEVER `wrong`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.scorer import (  # noqa: E402
    is_pure_json,
    normalize,
    score,
    score_full,
    try_extract_json,
)

CANBERRA = {"canonical_answer": "Canberra", "acceptable_variants": [], "banned_word": "Canberra"}
NUM56 = {"canonical_answer": "56", "acceptable_variants": ["fifty-six"], "banned_word": "56"}
BRASILIA = {"canonical_answer": "Brasilia", "acceptable_variants": ["brasília"], "banned_word": "Brasilia"}


def test_prose_substring_correct():
    assert score("The capital of Australia is Canberra.", "prose", CANBERRA) == "correct"


def test_prose_wrong():
    assert score("It is Sydney, I believe.", "prose", CANBERRA) == "wrong"


def test_json_plain_correct():
    assert score('{"answer": "Canberra"}', "json", CANBERRA) == "correct"


def test_json_code_fence_correct():
    assert score('```json\n{"answer": "Canberra"}\n```', "json", CANBERRA) == "correct"


def test_json_trailing_comma_correct():
    assert score('{"answer": "Canberra",}', "json", CANBERRA) == "correct"


def test_json_nested_answer_correct():
    assert score('{"result": {"answer": "Canberra"}}', "json", CANBERRA) == "correct"


def test_json_wrapped_in_chatter_is_still_extractable():
    # lenient extraction fishes the object out -> gradeable (correct), though NOT pure.
    txt = 'Sure! Here is your JSON: {"answer": "Canberra"}. Hope that helps!'
    assert score(txt, "json", CANBERRA) == "correct"


def test_json_no_object_is_unparseable_not_wrong():
    assert score("The answer is Canberra.", "json", CANBERRA) == "unparseable"


def test_json_object_without_answer_key_is_unparseable():
    assert score('{"city": "Canberra"}', "json", CANBERRA) == "unparseable"


def test_json_wrong_answer_is_wrong_not_unparseable():
    assert score('{"answer": "Sydney"}', "json", CANBERRA) == "wrong"


def test_number_as_string_and_as_number_both_match():
    assert score('{"answer": "56"}', "json", NUM56) == "correct"
    assert score('{"answer": 56}', "json", NUM56) == "correct"


def test_accent_folding_matches():
    assert score("The capital is Brasília.", "prose", BRASILIA) == "correct"
    assert score('{"answer": "Brasilia"}', "json", BRASILIA) == "correct"


def test_pure_json_compliance():
    assert is_pure_json('{"answer": "Canberra"}') is True
    assert is_pure_json('```json\n{"answer": "Canberra"}\n```') is False
    assert is_pure_json('Sure: {"answer": "Canberra"}') is False


def test_lexical_violation_tracked_but_correctness_independent():
    item = {"canonical_answer": "Canberra", "acceptable_variants": [], "banned_word": "Canberra"}
    # used the banned word AND is correct -> correct bucket, violation flagged
    r = score_full("The answer is Canberra.", "lexical", item)
    assert r["bucket"] == "correct"
    assert r["lexical_violation"] is True
    # complied (no banned word) but still conveys the answer via variant-free prose
    r2 = score_full("Australia's seat of government is that planned inland city.", "lexical", item)
    assert r2["lexical_violation"] is False


def test_json_answer_with_units_credits_the_number():
    m18 = {"canonical_answer": "18", "acceptable_variants": [], "banned_word": "18"}
    assert score('{"answer": "$18"}', "json", m18) == "correct"
    assert score('{"answer": "18 dollars"}', "json", m18) == "correct"
    assert score('{"answer": "180"}', "json", m18) == "wrong"          # 18 not a token in 180
    pct = {"canonical_answer": "60", "acceptable_variants": [], "banned_word": "60"}
    assert score('{"answer": "60%"}', "json", pct) == "correct"
    n108 = {"canonical_answer": "108", "acceptable_variants": [], "banned_word": "108"}
    assert score('{"answer": 104}', "json", n108) == "wrong"           # genuinely wrong stays wrong


def test_word_boundary_avoids_substring_false_positives():
    six = {"canonical_answer": "6", "acceptable_variants": ["six"], "banned_word": "6"}
    assert score("I have six apples.", "prose", six) == "correct"
    assert score("The answer is 6.", "prose", six) == "correct"
    assert score("There are sixty of them.", "prose", six) == "wrong"   # not 'six' in 'sixty'
    assert score("The total is 60.", "prose", six) == "wrong"           # not '6' in '60'


def test_normalize_basics():
    assert normalize("  Canberra! ") == "canberra"
    assert normalize('"Mexico City"') == "mexico city"


def test_try_extract_json_returns_none_on_prose():
    assert try_extract_json("just some words") is None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} scorer tests passed.")


if __name__ == "__main__":
    _run_all()
