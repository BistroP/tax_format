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

## 1. The dataset — mostly self-authored, plus a fetched GSM8K slice

Two sources, deliberately.

**Self-authored (350 items).** The justification is *the depth axis, and nothing else*:

1. **A controlled depth ladder.** The experiment is accuracy vs *reasoning depth* —
   4 → 40 sequential steps with everything else held constant. No fixed benchmark
   carries that axis; GSM8K problems are ~2–8 steps and aren't labelled by depth.
2. **Answer length held ~constant** (1–3 chars) at every rung, so the format signal
   isn't confounded by answer length — harder items must not have longer answers.
3. **Computed answer keys.** Multi-step answers are computed from the op chain (§1.3),
   then independently re-derived from the question *text* by `verify_stimuli`
   (291 items re-verified, 0 mismatches).
4. **Format-neutral question text**, identical across all conditions.

**Fetched (100 items, GSM8K).** The external control: our own chains can't be their
own check. [eval/make_gsm8k.py](../eval/make_gsm8k.py) pulls the official
[grade-school-math](https://github.com/openai/grade-school-math) test split (1,319
problems) and parses the `#### N` gold labels programmatically. Item ids carry the
source line — `gsm_0123` is line 123 of `test.jsonl` — so any item is traceable back
to the dataset. Two declared filters: answers restricted to **0–999** (preserves the
answer-length control above; costs 10% of the pool and incidentally removes every
comma-formatted label, which the scorer would fail to match against a canonical
`1000`), then the usual pilot gate (§5).

> **A retracted argument, kept visible.** An earlier draft justified self-authoring by
> claiming that a scraped dataset makes you "inherit its label errors." That is
> backwards, and we're not going to quietly delete it: GSM8K's labels are vetted and
> ours are not, so hand-authoring **adds** label risk rather than removing it. It also
> described the GSM8K slice as "externally vetted" while the items were in fact
> hand-transcribed by us — the claim leaned on our typing. Both are fixed above: the
> slice is now fetched, and the only defensible reason to self-author is the depth axis.

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

### 1.2 The ladder (main set: 450 items)

| category | n | source | tests |
|---|---|---|---|
| `factual_recall` | 20 | authored | capitals, elements, authors, dates — pure lookup, no reasoning |
| `arithmetic` | 10 | authored | one-step (`"7 × 8?"`) |
| `word_problem` | 10 | authored | one-step, framed in words |
| `two_hop` | 10 | authored | two chained facts ("language of the 2016 Olympics host?") |
| `ms_4step` / `ms_5step` / `ms_6step` | 100 each | authored | multi-step arithmetic chains |
| `gsm8k` | 100 | **fetched** | real grade-school word problems, gold labels |

The first 50 items are the easy band, and they earn their place by showing ~0 tax —
they're the evidence the instrument doesn't manufacture an effect. The 300 `ms_` items
carry the main result; the 100 GSM8K items are the external control.

Each `ms_` level is 50 hand-authored items topped up to 100 by
[eval/make_multistep.py](../eval/make_multistep.py). The authored items are ~86% bare
(`"Start with the number N…"`) and ~14% narrative (`"A theatre starts with 120 seats
booked…"`); the generator emits only the bare form, so each level sits at ~93% bare.
That surface-form shift is worth knowing, but it can't touch the tax, which is a
*within-item* prose-vs-constraint difference — every condition sees the same items.

A separate **600-item ceiling set** (8/12/16/20/28/40-step, 100 each) is used only for
the depth sweep (§8), ungated.

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
  confirm non-truncation (at 40 steps: 6/500 prose and 14/500 strict_schema hit the
  cap — against 404 that are simply *wrong*, so the ceiling is reasoning, not tokens).
- **On-disk caching**, keyed by `(model, item_id, condition)` **and validated against
  the stored prompt**. Payoffs: the run is reproducible; re-running is nearly free; and
  **re-scoring cached text after a scorer change costs zero API calls**. A parallel
  `prefetch()` (thread pool) warms the cache so a ~9,000-call sweep finishes in ~20
  minutes. Transient OpenRouter failures (a 429 wrapped as a provider 400 after
  fail-over) are retried; genuine bad requests are not.

Each cached record stores `prompt`, `text`, `stop_reason`, and `latency_s`.

> **Why the prompt check exists.** The id is not the question. Raising the sweep's
> `PER_LEVEL` reshuffles the generator's stream, so `deep12_00` keeps its id and gets a
> *different question* — and an id-keyed cache would hand back the old answer for the
> new question, silently, with no error anywhere. `query()` therefore builds the prompt
> *before* the lookup and treats a prompt mismatch as a plain miss. Content-addressing
> the cache is what makes "just bump n" a safe operation rather than a data-corruption
> bug. (The per-level RNG in §8 attacks the same problem from the other end.)

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

4. **The depth curve** — accuracy vs. n-step for prose / json / strict_schema (§8),
   in the explorer and as a standalone `ceiling.html`. **Every point is clickable**:
   it lazy-loads `ceiling/level_NN.json` and lists all 100 questions at that depth with
   each of the 5 models' responses, bucket-tagged, click-to-expand. Shipping all six
   levels up front would be ~5MB for a chart most visitors only glance at, so the
   per-level files are fetched on demand.

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
- **Depths 8 / 12 / 16 / 20 / 28 / 40** (100 items each = 600 items), generated the
  same way (§1.3). Each level draws from its own seeded RNG (`SEED + level`) so raising
  `PER_LEVEL` *appends* items instead of reshuffling existing ids.
- **Ungated** — no pilot filter; failures are the point.
- **`max_tokens = 4096`** so long reasoning isn't truncated (the critical control —
  otherwise the "ceiling" is a token limit).
- Conditions: `prose`, `json`, `strict_schema`.

Each point is 100 questions × 5 models = **500 responses**.

| steps | `prose` | `json` | `strict_schema` |
|---|---|---|---|
| 8 | 100% | 2% | 96% |
| 12 | 99% | 1% | 93% |
| 16 | 98% | 1% | 81% |
| 20 | 97% | 1% | 71% |
| 28 | 95% | 1% | 42% |
| 40 | 92% | 1% | **16%** |

- `json` is a **floor** — ~1% throughout (no scratchpad → ~1–2 effective steps).
- `prose` is a **high, model-stratified ceiling** — GPT-4o and Mistral 24B hold 100% at
  40 steps, Gemma 97%, Llama 70B 95%; only Llama 8B clearly bends (100% → 68%). The
  frontier ceiling is past 40 for these bounded chains.
- `strict_schema` **decays with depth**, 96% → 16%, and the decay is *genuine reasoning
  failure*: at 40 steps it's 404 `wrong` vs 17 `unparseable` (valid JSON, filled-in
  reasoning field, wrong answer), with only 14/500 hitting the token cap.

### 8.1 The mechanism: a `reasoning` field is not a scratchpad

Measuring how much reasoning each format actually *elicits* (median chars — for `prose`
the whole response, for `strict_schema` the contents of its reasoning field):

| steps | prose | strict reasoning field | ratio | prose acc | strict acc |
|---|---|---|---|---|---|
| 8 | 362 | 212 | 0.58 | 100% | 96% |
| 16 | 702 | 285 | 0.41 | 98% | 81% |
| 20 | 853 | 338 | 0.40 | 97% | 71% |
| 28 | 1178 | 197 | 0.17 | 95% | 42% |
| 40 | 1658 | **106** | **0.06** | 92% | **16%** |

In prose, reasoning **scales with the problem** — 362 → 1,658 chars as depth grows 8 →
40. Under the schema it grows to 338 chars at 20 steps and then **collapses to 106** —
*less* reasoning for a *harder* problem — and accuracy tracks it exactly. The field
stops behaving like a scratchpad and starts behaving like a **budget**: the model fills
it with a token gesture and commits to an answer it hasn't worked out.

The clearest single artifact in the project — GPT-4o, same 40-step question
(`deep40_000`, gold answer **51**), the two formats side by side:

**`prose` → correct.** 1,701 chars; every step numbered and evaluated:

```
1. Start with **88**.
2. Divide by 2: \( 88 \div 2 = 44 \).
3. Subtract 21: \( 44 - 21 = 23 \).
   … all 40 steps … → 51
```

**`strict_schema` → wrong.** 179 chars, the entire response:

```json
{"result": {"reasoning": "The operations were performed step by step as described in
the question, ensuring accuracy at each step.", "answer": "105", "confidence": 1}}
```

The reasoning field contains a *description of having reasoned* and not one digit of
arithmetic. The model asserts the process it did not perform, answers 105, and reports
`confidence: 1`. This is what the 0.06 ratio in the table looks like from the inside —
and it's why the failure lands in `wrong` rather than `unparseable`: the JSON is
flawless. A validator would pass this response. Every schema-conformance metric in the
industry would call it a success.

The novel claim: the format tax is **not a fixed penalty but a depth-dependent one**.
"Add a `reasoning` field" genuinely recovers the tax at 4–6 steps (§6) — which is why
it's standard advice — and then **silently stops working** as depth grows. This goes
past "reasoning recovers the loss" (Tam) to "the recovery itself has a depth limit,"
and past "capacity competition" (Fan) by measuring the capacity the format concedes.

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
- **Bounded 2–3-digit arithmetic** — isolates *depth* (per-step math is easy), but
  doesn't test hard per-step computation.
- **The strong-model prose ceiling is not located** — it is beyond 40 steps.
- **One prompt per condition** — the tax is measured against *our* phrasing of "answer
  in JSON"; we don't sweep prompt variants, so some of the effect may be promptable.
- **GSM8K answers filtered to 0–999** (§1) to hold answer length constant — that's 90%
  of the split, but it is a declared deviation from "GSM8K" as published.
- **`ms_` levels mix authored and generated phrasings** (~93% bare / ~7% narrative).

---

## Appendix — file map

```
data/stimuli.json          the 450-item stimulus set (350 authored + 100 fetched GSM8K)
data/piloted_stimuli.json  those that passed the prose gate
eval/config.py             models, conditions, settings (temperature, max_tokens, "reasoning off")
eval/conditions.py         the four prompt templates
eval/harness.py            OpenRouter calls, prompt-checked caching, retry, parallel prefetch
eval/scorer.py             the three-bucket scorer + compliance flags
eval/numwords.py           integer -> word spellings (acceptable_variants)
eval/make_gsm8k.py         fetch the official GSM8K test split -> the gsm8k slice
eval/make_multistep.py     top the ms_4/5/6step levels up to n=100
eval/verify_stimuli.py     re-derive every arithmetic answer from its question text
eval/pilot.py              the prose gate
eval/run_eval.py           run + score + aggregate -> results.json
eval/make_heatmap.py       static heatmap.html
eval/ceiling_sweep.py      the ungated depth sweep -> ceiling.json + ceiling/level_*.json
eval/list_models.py        browse/verify OpenRouter model slugs
tests/test_scorer.py       offline scorer unit tests (no API key)
web/index.html             the interactive explorer
web/server.py              FastAPI backend (/results.json, /live)
results.json               the computed benchmark (the explorer reads this)
ceiling.json / ceiling.html  the depth-sweep data + chart
```
