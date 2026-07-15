# Methodology & Code Walkthrough

How the Format-Tax Explorer is built, end to end — the dataset, the conditions, the
model harness, the scorer, the metrics, and the visualization. This is the "due
diligence" document: enough detail to defend every number.

For the headline results, see [README.md](../README.md). This document is about
*how* those numbers are produced.

---

## 0. What kind of artifact this is

This is a **benchmark / controlled measurement instrument**, not a survey. A survey
reviews other people's results; here we *construct* a stimulus set, a scorer, and a
harness, and *run* them to produce new measurements. It also has a small
**replication** element — re-testing a documented effect (the "format tax", Tam et
al. 2024) on current (2026) models across multiple vendors.

The "format tax" premise: constraining an LLM to a rigid output format (e.g. JSON)
degrades its *reasoning*, not just its formatting — because reasoning and formatting
draw on the same token-by-token generation budget. This matters in practice because
structured output is ubiquitous in agentic systems, tool/function calling, and APIs.

---

## 1. The dataset — self-authored, not off the shelf

We authored every stimulus. No GSM8K, no MMLU. Five reasons this is deliberate:

1. **Exact, machine-checkable answers.** Every item resolves to one short token — a
   name, a year, a number — so grading is a token match, not fuzzy prose comparison.
2. **Guaranteed-correct answer keys.** Multi-step answers are *computed from* the
   problem (§1.3), so the key cannot be wrong. Scraped datasets carry label noise.
3. **A controlled difficulty ladder.** We need items from trivial to straining,
   otherwise held constant — you can't get a clean 4→40-step depth axis from a fixed
   benchmark.
4. **Answer length held ~constant**, so the format-overhead signal isn't confounded
   by answer length (harder items must not have longer answers).
5. **Format-neutral question text**, identical across all conditions.

### 1.1 Item schema

Every stimulus is one object with six fields ([data/stimuli.json](../data/stimuli.json)):

```json
{
  "id": "ms6_01",
  "question": "Start with the number 10. Add 15, then multiply by 2, then subtract 8, then divide by 6, then add 20, then multiply by 4. What is the result?",
  "canonical_answer": "108",
  "acceptable_variants": ["one hundred eight", "one hundred and eight"],
  "category": "ms_6step",
  "banned_word": "108"
}
```

- `canonical_answer` — the one correct short answer.
- `acceptable_variants` — alternate surface forms that should also count as correct:
  number word-forms (`"fifty-six"` / `"fifty six"`), accent/spelling alternates
  (`"brasília"`, `"dostoyevsky"`), fuller names (`"george orwell"`). These prevent
  the scorer from marking a *right* answer *wrong* over phrasing.
- `category` — the ladder rung.
- `banned_word` — the token the `lexical` condition forbids (set to the answer).

### 1.2 The ladder (main set: 110 items, 106 kept after the pilot gate)

| category | n | tests |
|---|---|---|
| `factual_recall` | 20 | capitals, elements, authors, dates — pure lookup, no reasoning |
| `arithmetic` | 10 | one-step (`"7 × 8?"`) |
| `word_problem` | 10 | one-step, framed in words |
| `two_hop` | 10 | two chained facts ("language of the 2016 Olympics host?") |
| `ms_4step` / `ms_5step` / `ms_6step` | 20 each | multi-step arithmetic chains |

A separate **72-item ceiling set** (8/12/16/20/28/40-step, 12 each) is used only for
the depth sweep (§8).

### 1.3 Computed answers (multi-step)

Multi-step arithmetic is generated, not hand-written. A generator picks a start value
and a sequence of operations, keeping every intermediate a positive integer in
[10, 999], and **the prose and the answer are emitted from the same operation list**:

```python
v = start
for op, k in ops:            # e.g. ("add", 15), ("mul", 2), ("div", 6), ...
    v = v + k if op == "add" else v - k if op == "sub" else v * k if op == "mul" else v // k
answer   = v                 # the key IS the computed result
question = render(start, ops)  # "...Add 15, then multiply by 2, ..." from the SAME ops
```

Divisions are only chosen when the value divides evenly, so "divide by 6" is exact.
A model that follows the wording *is* doing exactly the computation that produced the
key. An independent recompute + `assert` double-checks each generated item.

**You don't have to trust the keys.** `python -m eval.verify_stimuli`
([eval/verify_stimuli.py](../eval/verify_stimuli.py)) re-parses every arithmetic
question's *text* back into operations and recomputes, independently of the
generator, and exits non-zero on any mismatch (currently: 51 items re-verified, 0
mismatches; the remaining hand-authored factual / word-problem / two-hop items are
pilot-gated).

---

## 2. The conditions — "format is the only variable"

One question, four instructions. The question string is byte-identical across all
four, so any accuracy difference is *caused by the instruction*. From
[eval/conditions.py](../eval/conditions.py):

```
prose:         "Answer the question.\n\nQuestion: {q}"

json:          'Respond ONLY with a JSON object of the form {"answer": <your answer>}
                and nothing else.\n\nQuestion: {q}'

strict_schema: 'Respond ONLY with a JSON object matching this schema and nothing else:
                {"result": {"reasoning": <string>, "answer": <string>,
                            "confidence": <number 0-1>}}\n\nQuestion: {q}'

lexical:       'Answer the question without using the word "{banned}".\n\nQuestion: {q}'
```

What each isolates:

| condition | format constrained? | reasoning allowed? |
|---|---|---|
| `prose` | no | yes — the control |
| `json` | yes | **no** ("nothing else") |
| `strict_schema` | yes (nested, more required fields) | **yes** (a `reasoning` field, first) |
| `lexical` | no (one output token banned) | yes |

The `json` vs `strict_schema` contrast is load-bearing: `strict_schema` is *stricter*
yet scores *higher*, which pins the cause to reasoning-suppression rather than
format-rigidity.

---

## 3. The model harness

[eval/harness.py](../eval/harness.py). Everything routes through **OpenRouter** via
the OpenAI SDK — one key, five models, four vendors, so no vendor grades its own
homework.

**The five models** (all non-reasoning instruct models):

| model | vendor | tier |
|---|---|---|
| Llama 3.1 8B | Meta | small open |
| Mistral Small 24B | Mistral | mid open |
| Gemma 3 27B | Google | mid open |
| Llama 3.3 70B | Meta | large open |
| GPT-4o | OpenAI | closed frontier |

Four settings matter for rigor:

- **`temperature = 0`** where the model allows it — stabilizes the replication.
- **Reasoning off.** No thinking/reasoning parameter is sent, and only instruct
  models are used. A model that reasons internally before formatting would recover
  the loss and *hide* the tax. (Gemini 3.5 Flash was excluded for exactly this — it
  returns `400: "reasoning is mandatory"` and cannot be run non-reasoning; Gemma 3
  27B is Google's open instruct model and behaves like the others.)
- **`max_tokens`.** 512 for the main run; **4096 for the depth sweep**, because a
  40-step solution written longhand is ~1,000 tokens and a 512 cap would *truncate*
  the reasoning and produce a fake ceiling. `stop_reason` is checked afterward to
  confirm non-truncation (only 2/60 responses hit the cap at 40 steps).
- **On-disk caching**, keyed by `(model, item_id, condition)`. Payoffs: the run is
  reproducible; re-running is nearly free; and **re-scoring cached text after a
  scorer change costs zero API calls**. A parallel `prefetch()` (thread pool) warms
  the cache so a ~1,000-call run finishes in minutes. Transient OpenRouter failures
  (a 429 wrapped as a provider 400 after fail-over) are retried; genuine bad requests
  are not.

Each cached record stores `text`, `stop_reason`, and `latency_s`.

---

## 4. The scorer — the heart of the project

[eval/scorer.py](../eval/scorer.py). Every response is bucketed into exactly one of
**`correct` / `wrong` / `unparseable`**, with two *independent* compliance flags
alongside. The single most important rule:

> **A parse failure is never counted as a wrong answer.** `unparseable` is its own
> bucket.

### 4.1 Normalization (applied to everything)

```python
def normalize(s):
    s = unicodedata.normalize("NFKD", s)                       # decompose accents
    s = "".join(c for c in s if not unicodedata.combining(c))  # drop accent marks
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()                         # collapse whitespace
    s = s.strip(_EDGE_PUNCT)                                    # strip surrounding punct
    return s
```

`"Brasília."` → `brasilia`; `'"Mexico City"'` → `mexico city`. The accepted set for
an item is `{normalize(canonical)} ∪ {normalize(variant) for each variant}`.

### 4.2 Two grading paths

**Path A — `prose` / `lexical` (free-form text):** does an accepted answer appear as
a *whole token* in the normalized response?

```python
norm = normalize(response_text)
return "correct" if any(_word_in(a, norm) for a in accepted) else "wrong"
```

`_word_in` is a whole-token (not raw substring) test, because number-words are
substrings of each other (`"six"` ⊂ `"sixty"`, `"6"` ⊂ `"60"`). It uses lookarounds
rather than `\b` so multi-word answers still match:

```python
re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack)
```

`"six"` matches "I have six apples" but **not** "sixty"; `"6"` doesn't match "60";
`"mexico city"` still matches as a phrase.

**Path B — `json` / `strict_schema`:** extract the JSON, read the `answer` field, and
whole-token-match the accepted answer *inside* it.

```python
parsed = try_extract_json(response_text)
answer = _extract_answer_field(parsed)     # top-level "answer", or nested one level deep
if answer is None:
    return "unparseable"                   # <-- the whole point
candidate = normalize(str(answer))
return "correct" if any(_word_in(a, candidate) for a in accepted) else "wrong"
```

The difference from Path A is *scope*, not strictness: prose matches the answer as a
token anywhere in the full response; the JSON path matches only inside the extracted
`answer` field (the reasoning field and surrounding text don't count). Whole-token
matching lets a units-bearing answer — `"$18"`, `"3 bolts"`, `"60%"` — credit the value
`18` / `3` / `60` (GSM8K answers carry units), while a wrong `"104"` still fails against
`"108"` and `"18"` won't match inside `"180"`. (This path originally used exact string
equality; GSM8K's unit-bearing answers exposed that as too strict, and re-scoring from
cache fixed it at zero API cost — which is exactly why we cache raw responses.)

### 4.3 The lenient JSON extractor

`try_extract_json` tolerates the common messes, in order:

1. **Strip markdown code fences** — ` ```json … ``` ` → inner text. (GPT-4o and Gemma
   fence everything.)
2. **Find the first balanced `{…}` block** with a scanner that tracks string state and
   escapes, so a `}` inside a string doesn't prematurely close the object.
3. **`json.loads`**; on failure, **strip trailing commas** (`,}` → `}`) and retry once.
4. Return the parsed object, or `None`.

`_extract_answer_field` then looks for `"answer"` at the top level, or one level deep
(so `strict_schema`'s `{"result": {"answer": …}}` resolves). If neither yields a
value → `unparseable`.

### 4.4 What happens to `unparseable`

A response is `unparseable` when:
- there's **no JSON object at all** (the model answered in prose despite the
  instruction), or
- there's an object but **no `answer` key** (`{"city": "Canberra"}`), or
- the braces are there but the content is **not valid JSON even after the lenient
  fixes** — e.g. Llama 70B's `{"answer": (((10+15)*2 - 8)/6 ...)}`: `json.loads`
  chokes on a bare arithmetic expression as a value, no trailing comma to fix, so
  extraction returns `None`.

It then stays separate at every level:
- **Scoring:** returned as its own string, never `"wrong"`.
- **Aggregation:** `unparseable_rate` is its own column, never merged into
  `wrong_rate`.
- **UI:** a visually distinct dashed-purple badge, separate from red "wrong".
- **Conceptually:** an unparseable response is a **format-compliance** failure, not a
  **reasoning** failure. Folding it into "wrong" would inflate the tax with
  JSON-syntax noise and destroy the ability to tell "reasoned wrong" from "couldn't
  emit valid JSON." A high unparseable rate is itself a finding — Llama 70B: 18% of
  its JSON is malformed, a real, reportable weakness distinct from its reasoning.

### 4.5 Two independent compliance flags (`score_full`)

Correctness is one axis; *format compliance* is a second, tracked separately:

- **`pure_json_compliance`** (`is_pure_json`): did the **entire, raw** response parse
  as JSON, with no surrounding prose or fences? Stricter than "could I fish JSON out
  of it." A fenced `` ```json {"answer": "Canberra"} ``` `` is **correct**
  (extractable) but **not pure** — it failed the *format* while succeeding at the
  *reasoning*. GPT-4o: ~100% correct, ~7% pure-JSON, because it code-fences.
- **`lexical_violation`** (`contains_banned_word`): in the `lexical` condition, did
  the response use the forbidden word anyway? A compliance number, independent of
  correctness.

### 4.6 Worked examples (from the scorer's self-test)

| response | condition | bucket |
|---|---|---|
| `The capital is Canberra.` | prose | correct |
| `` ```json\n{"answer": "Canberra"}\n``` `` | json | correct (fence stripped) |
| `I think it is Sydney.` | prose | wrong |
| `Here you go, hope it helps!` | json | **unparseable** |
| `{"answer": 56}` / `{"answer": "56"}` | json | both correct (number-as-string normalized) |
| `{"city": "Canberra"}` | json | **unparseable** (no `answer` key) |

The scorer has an offline unit-test suite ([tests/test_scorer.py](../tests/test_scorer.py))
covering every edge case above — it runs without any API key.

---

## 5. The pilot gate

Before use, [eval/pilot.py](../eval/pilot.py) runs each item in **prose across all
five models** and keeps it **only if every model gets it right unconstrained.** The
tax is a *drop* from prose, so prose must first be a real ceiling; a low JSON score on
a problem the model can't do anyway tells you nothing about *format*. Four items were
dropped (`wp_003`, `twh_010`, `ms6_02`, `ms6_05`).

Consequences:
- In the main run, the prose column is **100% by construction** — the gate working,
  not luck.
- The **depth sweep is deliberately ungated** (§8), because its purpose is to find
  where prose *itself* breaks. Gating it would discard exactly the failures we hunt.

---

## 6. Metrics and the "tax"

[eval/run_eval.py](../eval/run_eval.py). For every `(model, condition)`, over the
graded items:

- `correct_rate`, `wrong_rate`, `unparseable_rate` (these three partition 100%),
- `pure_json_compliance`, `lexical_violation_rate` (the separate compliance signals),

computed both overall and **per category** (`by_category`, which powers the difficulty
filter). The headline number:

```
tax(model, condition) = correct_rate(prose) − correct_rate(condition)
```

Everything lands in `results.json`: `meta` (models, conditions, settings, timestamp),
`items`, a `responses` map keyed `"model|item|condition" → {text, bucket, pure_json,
…}`, `metrics`, `by_category`, `tax`. Because grading reads cached text, regenerating
`results.json` after a scorer change is instant and free.

---

## 7. The visualization

Two front-end artifacts:

- **`heatmap.html`** — a static, self-contained colored table (`make_heatmap.py`). No
  JavaScript, no interactivity. The "drop it in a slide / works offline" fallback.
- **The interactive explorer** — [web/index.html](../web/index.html) served by a tiny
  FastAPI backend ([web/server.py](../web/server.py)); reads `results.json`. Launch
  with `python -m uvicorn web.server:app` → `http://127.0.0.1:8000`.

The explorer has three layers:

1. **Heatmap.** Rows = models, columns = conditions; each cell shows accuracy,
   **colored by the tax** (green = none → red = heavy), with compliance sub-labels
   (`unparseable 18%`, `pure-JSON 0%`, `banned word 67%`). A dropdown **filters by
   category**: `factual_recall` is all green; `ms_6step` turns the JSON column red.
2. **Click-to-drill.** Clicking a cell lists every question behind it — the question,
   the canonical answer, and the model's **prose answer vs. its answer under that
   condition, side by side**, each tagged `correct` / `wrong` / `unparseable`
   (unparseable styled distinctly), with correct→not-correct rows highlighted.
3. **Live panel.** A "try it yourself" box → `POST /live` → runs the question through
   the **same harness and scorer** as the offline pipeline → a real-time grid. A dead
   key degrades to an inline error chip.

Plus the **depth curve** (`ceiling.html`) — accuracy vs. n-step for prose / json /
strict_schema (§8).

Design invariant: every visual is a **view over `results.json`** (or `ceiling.json`);
the front-end never computes anything the scorer didn't already decide.

---

## 8. The reasoning-ceiling sweep (second experiment)

[eval/ceiling_sweep.py](../eval/ceiling_sweep.py). This reframes the project as **two
axes** instead of one:

- **Format tax** — `prose − json`, from the main run.
- **Reasoning ceiling** — prose accuracy vs. step-depth, *ungated*, to find where
  chain-of-thought itself fails.

Design differences from the main run:
- **Depths 8 / 12 / 16 / 20 / 28 / 40** (12 items each), generated the same way (§1.3).
- **Ungated** — no pilot filter; failures are the point.
- **`max_tokens = 4096`** so long reasoning isn't truncated (the critical control —
  otherwise the "ceiling" is a token limit).
- Conditions: `prose`, `json`, `strict_schema`.

Findings (averaged over the five models):
- `json` is a **floor** — ~0% past 8 steps (no scratchpad → ~1–2 effective steps).
- `prose` is a **high, model-stratified ceiling** — 4 of 5 models hold ~100% to 40
  steps; only Llama 8B clearly bends (100% → 50% by 40). The frontier ceiling is past
  40 for these bounded chains.
- `strict_schema` **decays with depth** — ~92% at 8 steps → ~22% at 40 — and it is
  *genuine reasoning failure* (at 40 steps: 45/60 `wrong` with valid JSON, only 1–2
  truncated, avg 646 chars, far below the 4096-token cap). The same GPT-4o that writes
  ~1,900 chars of working in prose emits a ~198-char reasoning field under the schema
  and answers wrong.

The novel claim: the format tax is **not a fixed penalty but a depth-dependent one** —
structured output progressively starves reasoning as problems deepen, even when a
reasoning field is nominally provided. This goes past "reasoning recovers the loss"
(Tam) to "the recovery itself has a depth limit."

---

## 9. Threats to validity

**Controlled:**
- *Answer length* held ~constant, so the tax isn't length in disguise.
- *Reasoning mode off* (instruct-only), so reasoning and formatting genuinely compete.
- *Multi-vendor* (Meta / Mistral / Google / OpenAI) — not one lab's quirk; no
  self-grading.
- *Guaranteed-correct keys* via computed answers.
- *Format vs. reasoning* separated by the three-bucket scorer.
- *`max_tokens` verified* non-truncating on the depth sweep.

**Acknowledged limits:**
- **Exact-match, no partial credit** — one slip in a 40-step chain is a full miss, so
  multi-step numbers are a *worst-case* read.
- **`lexical` correctness is compliance-confounded** — banning the answer word means a
  compliant reply omits the only string the scorer can match, so read `lexical` as a
  compliance signal (violation rate), not a reasoning tax.
- **n = 12–20 per cell** — demo-grade error bars, not publication-grade.
- **Bounded 2–3-digit arithmetic** — isolates *depth* (per-step math is easy), but
  doesn't test hard per-step computation.
- **The strong-model prose ceiling is not located** — it is beyond 40 steps.

---

## Appendix — file map

```
data/stimuli.json          the 110-item stimulus set (self-authored)
data/piloted_stimuli.json  the 106 that passed the prose gate
eval/config.py             models, conditions, settings (temperature, max_tokens, "reasoning off")
eval/conditions.py         the four prompt templates
eval/harness.py            OpenRouter calls, caching, retry, parallel prefetch
eval/scorer.py             the three-bucket scorer + compliance flags
eval/pilot.py              the prose gate
eval/run_eval.py           run + score + aggregate -> results.json
eval/make_heatmap.py       static heatmap.html
eval/ceiling_sweep.py      the ungated depth sweep -> ceiling.json + ceiling.html
eval/list_models.py        browse/verify OpenRouter model slugs
tests/test_scorer.py       offline scorer unit tests (no API key)
web/index.html             the interactive explorer
web/server.py              FastAPI backend (/results.json, /live)
results.json               the computed benchmark (the explorer reads this)
ceiling.json / ceiling.html  the depth-sweep data + chart
```
