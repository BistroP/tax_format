"""Format conditions (Section 5.2). Same question across all conditions; only the
instruction changes. That invariance is the whole point — format is the only variable."""

from __future__ import annotations

# A more rigid / nested schema, to show the tax scaling with schema complexity (per Fan).
STRICT_SCHEMA_HINT = (
    '{"result": {"reasoning": <string>, "answer": <string>, "confidence": <number 0-1>}}'
)


def build_prompt(condition: str, item: dict) -> str:
    q = item["question"]
    if condition == "prose":
        return f"Answer the question.\n\nQuestion: {q}"
    if condition == "json":
        return (
            'Respond ONLY with a JSON object of the form {"answer": <your answer>} '
            f"and nothing else.\n\nQuestion: {q}"
        )
    if condition == "lexical":
        banned = item.get("banned_word", "")
        return f'Answer the question without using the word "{banned}".\n\nQuestion: {q}'
    if condition == "strict_schema":
        return (
            "Respond ONLY with a JSON object matching this schema and nothing else:\n"
            f"{STRICT_SCHEMA_HINT}\n\nQuestion: {q}"
        )
    raise ValueError(f"unknown condition: {condition}")


# Human-readable descriptions for the UI / README.
CONDITION_LABELS = {
    "prose": "Prose (control)",
    "json": "JSON",
    "lexical": "Lexical ban",
    "strict_schema": "Strict schema",
}
