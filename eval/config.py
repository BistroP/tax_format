"""Central configuration for the eval pipeline (Section 5.4)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"

STIMULI_PATH = DATA / "stimuli.json"
PILOTED_PATH = DATA / "piloted_stimuli.json"
RESULTS_PATH = ROOT / "results.json"

# --- Model harness (Section 5.4) --------------------------------------------
# Everything routes through OpenRouter (one key, many providers) so the
# replication never leans on a single vendor grading its own homework. The
# spread IS the story: a weak open model pays the format tax; a frontier model
# may absorb it. Slugs verified against OpenRouter's live catalog.
#
# Deliberately INSTRUCT (non-reasoning) models only — no o1/o3/R1/reasoning
# models. Same rationale as keeping "thinking" off: the tax is capacity
# competition (Fan 2026); a reasoning model reasons-first-formats-later and
# recovers most of the loss, masking the effect. temperature=0 for stability.
MODELS = [
    {"key": "llama8b",   "provider": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct",        "label": "Llama 3.1 8B (Meta)",         "temperature": 0},
    {"key": "mistral24b","provider": "openrouter", "model": "mistralai/mistral-small-24b-instruct-2501","label": "Mistral Small 24B (Mistral)", "temperature": 0},
    # Google's open instruct model. (Gemini 3.5 Flash can't run non-reasoning — 400 "reasoning
    # is mandatory" — so it would reason-first and mask the tax; Gemma keeps the control clean.)
    {"key": "gemma27b",  "provider": "openrouter", "model": "google/gemma-3-27b-it",                    "label": "Gemma 3 27B (Google)",        "temperature": 0},
    {"key": "llama70b",  "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct",        "label": "Llama 3.3 70B (Meta)",        "temperature": 0},
    {"key": "gpt4o",     "provider": "openrouter", "model": "openai/gpt-4o-2024-11-20",                 "label": "GPT-4o (OpenAI)",             "temperature": 0},
    # Other verified-reliable slugs to swap in: deepseek/deepseek-chat-v3-0324.
    # Avoid qwen/qwen-2.5-72b-instruct — its OpenRouter providers 400/429'd during testing.
    # Free-tier variants exist (rate-limited):  python -m eval.list_models :free
]

# Conditions (Section 5.2). MVP runs prose + json; the rest are stretch.
MVP_CONDITIONS = ["prose", "json"]
ALL_CONDITIONS = ["prose", "json", "lexical", "strict_schema"]

MAX_TOKENS = 512

# Reasoning is never enabled: the model set is instruct-only and no reasoning
# param is sent. This is methodological, not incidental — the format tax is a
# capacity-competition effect (Fan 2026). A reasoning model reasons-first-
# formats-later, recovering most of the loss and MASKING the tax. Instruct-only
# forces reasoning and formatting to share one generation budget.
REASONING = "disabled (instruct-only models, no reasoning param)"
