# Format-Tax Explorer

An interactive instrument that makes the known **"format tax"** visible and
independently re-tests it on current (2026) models.

> **The finding is not ours.** Constraining an LLM to a rigid output format (JSON,
> a banned word) degrades its *reasoning* accuracy — established by Tam et al.
> (2024) and mechanistically attributed to *capacity competition* by Fan (2026).
> What we built is the **instrument**, an **independent replication** on current
> models, and a **scorer** honest enough to tell "the model was wrong" apart from
> "I couldn't parse it."

**One-liner:** *The format tax is an established effect; this is an interactive
instrument that makes it visible and independently re-tests whether it still holds
on today's models — including the honest question of whether some frontier models
now absorb the tax while weaker ones still pay it.*

---

## What we claim (and what we don't)

**We claim:** (i) an explorer that turns a table-bound result into something you can
see and probe; (ii) an independent, multi-vendor replication with an independent
scorer; (iii) a scorer that separates **reasoning failure** from **parse failure**;
and (iv) one thing we haven't found stated elsewhere — **the tax is depth-dependent,
and the standard fix has a depth limit**. "Give the schema a `reasoning` field" fully
recovers the tax at 4–6 steps and collapses to 16% by 40 steps, because the field stops
scaling with the problem (338 chars of reasoning at 20 steps → 106 at 40). If (iv) is
already in the literature, it's (i)–(iii) that stand.

**We do NOT claim** novelty of the effect (→ Tam 2024), of capacity competition as the
mechanism (→ Fan 2026), or cross-library comparison (→ Geng et al.). See
[Citations](#citations).

---

## What this run shows

Reproduced across **5 instruct models / 4 vendors** (Llama 3.1 8B · Mistral 24B · Gemma 3 27B ·
Llama 3.3 70B · GPT-4o, via OpenRouter). Easy tasks: ~0 tax. Multi-step reasoning **collapses
under *answer-only* JSON — and recovers when the format permits reasoning.** That contrast
isolates the cause: it's chain-of-thought suppression, not format rigidity (Fan 2026).

Correct-rate on the multi-step items (prose baseline = 100% for every model by
construction — the pilot gate; n = 100 / 100 / 96):

| condition | what it does | 4-step | 5-step | 6-step |
|---|---|---|---|---|
| `prose` | free reasoning (control) | 100% | 100% | 100% |
| `json` | answer-only — reasoning **forbidden** | **6–41%** | **5–18%** | **1–9%** |
| `strict_schema` | rigid nested, but a `reasoning` field first | 95–100% | 94–100% | 88–100% |
| `lexical` | ban the answer token; reasoning allowed | ~94%\* | ~94%\* | ~90%\* |

**It reproduces on a real, externally-vetted set.** The **GSM8K** slice is *fetched* — 100 problems
sampled from the official [grade-school-math](https://github.com/openai/grade-school-math) test
split, gold labels parsed from the `#### N` line, ids traceable to the source file
([eval/make_gsm8k.py](eval/make_gsm8k.py)). On the 78 that pass the pilot gate, answer-only JSON
drops accuracy from 100% (prose) to **15–60%** (a −40 to −85pt tax); adding a `reasoning` field
recovers it to 44–97%. So the effect isn't an artifact of our self-authored chains — it holds on
problems written by someone else and labelled by someone else.

### And the recovery has a depth limit — the part we didn't expect

Push depth to 40 steps (600-item ungated sweep, 100 per level, `max_tokens=4096`) and the
"just add a `reasoning` field" fix **stops working**:

| steps | `prose` | `json` | `strict_schema` |
|---|---|---|---|
| 8 | 100% | 2% | 96% |
| 20 | 97% | 1% | 71% |
| 40 | **92%** | 1% | **16%** |

Not truncation (14/500 hit the cap, against 404 simply *wrong*) and not a parse failure (404
`wrong` vs 17 `unparseable` — **valid JSON, filled-in reasoning field, wrong answer**). The
mechanism is measurable: in prose, reasoning **scales with the problem** (362 → 1,658 median
chars from 8 → 40 steps); under the schema it grows to 338 chars at 20 steps then **collapses to
106** — *less* work on a *harder* problem — and accuracy tracks it exactly. **A `reasoning` field
is not a scratchpad; it's a budget.**

GPT-4o, same 40-step question, gold answer **51**. In prose it numbers all 40 steps across 1,701
chars and gets it right. Under the schema, this is the *entire* response:

```json
{"result": {"reasoning": "The operations were performed step by step as described in
the question, ensuring accuracy at each step.", "answer": "105", "confidence": 1}}
```

Not one digit of arithmetic — a *description* of having reasoned, a wrong answer, and
`confidence: 1`. The JSON is flawless, so every schema-conformance check in the industry passes
it. That's the point: **conformance metrics are blind to this failure by construction.**

Under `json` the models emit **bare, wrong numbers** (near-0% unparseable — genuine *reasoning*
failure, not a parse failure). `strict_schema` is *more* rigid yet **recovers** most of the loss
because it lets the model reason before the answer; `lexical` doesn't touch reasoning, so it stays
near prose. The tax tracks whether the format permits the scratchpad, not how strict it is.
Capability buys headroom **only at low difficulty** — on 4-step JSON, GPT-4o holds 41% while Llama
8B is at 6% — but by 5–6 steps **every model collapses to ≤18% and then ≤9%**, frontier included.

Two compliance signals, tracked separately from correctness: **Llama 3.3 70B emits *invalid* JSON
~29% of the time** (it writes the expression, e.g. `{"answer": ((10+15)*2…)}` — a *format* failure,
correctly bucketed `unparseable`, not `wrong`), while **Gemma and GPT-4o almost never emit *pure*
JSON** (0% / 2% — they wrap it in ```fences```) yet stay correct where they reason correctly. Note
how independent those two axes are: GPT-4o is near-worst on pure-JSON compliance and best on
accuracy. A conformance metric would rank these models almost backwards.

\* **`lexical` correctness is confounded — read it as *compliance*, not a reasoning tax.** Banning
the answer's canonical form means a faithfully-compliant answer omits the only string the scorer
can match. GPT-4o obeys the ban best (e.g. *"7 × 8 = 5 dozen minus 4"* — correct, but uncreditable),
so it scores *lowest* on `lexical`. That's the scorer's blind spot, not a reasoning failure; the
clean signal is the **banned-word-used rate** (shown per cell). Itself a finding: you can't cleanly
auto-grade a lexical constraint — it needs semantic/human eval.

## How it works

Two decoupled parts:

- **Offline eval pipeline (`eval/`, Python)** — the rigor layer. Authors → pilots →
  runs the benchmark → scores → writes `results.json`, caching every raw response.
- **Front-end (`web/`, static HTML + tiny backend)** — the demo: an interactive
  heatmap you drill into (per-item prose-vs-constraint, `unparseable` flagged
  distinctly from `wrong`), plus a live "try it yourself" panel. The static
  `heatmap.html` also stands alone.

### The three-bucket scorer (the load-bearing rigor)

Every response is bucketed as exactly one of **`correct` · `wrong` · `unparseable`**,
and `unparseable` is **never** folded into `wrong`. A model that can't emit valid
JSON X% of the time is failing the *format*, not the *reasoning* — that's its own
reportable number. We also track `pure_json_compliance` (did the *whole* response
parse, no chatter?) and `lexical_violation` (did it use the banned word?)
separately from correctness. Verified offline: `python tests/test_scorer.py`.

### Model spread (via OpenRouter — no vendor grades its own homework)

Routed through a single OpenRouter key so the replication spans providers instead
of leaning on any one vendor:

- **Llama 3.1 8B** (Meta) — weak, open
- **Mistral Small 24B** (Mistral) — mid, open
- **Gemma 3 27B** (Google) — mid, open
- **Llama 3.3 70B** (Meta) — large, open
- **GPT-4o** (OpenAI) — frontier, closed

The spread is the point: whether the weak open model pays the tax while a frontier
model absorbs it. Swap or widen in [eval/config.py](eval/config.py)
(`python -m eval.list_models` lists valid slugs).

### Methodological choices baked in

- **Pilot first.** Only items *all* models get right in prose survive — no baseline,
  no measurable drop.
- **Answer length held ~constant** (1–11 chars) so the tax isn't confounded by
  answer length.
- **Instruct-only models, on purpose.** No o1/o3/R1/reasoning models. The tax is
  capacity competition; a reasoning model reasons-first-formats-later, recovering
  most of the loss and *masking* the effect — so reasoning and formatting must
  share one generation budget.
- **`temperature=0`** for stability (every selected model accepts it).

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env         # then add your OPENROUTER_API_KEY

python tests/test_scorer.py  # offline, no key — proves the scorer (18 tests)

# build the stimulus set (both offline except the GSM8K fetch):
python -m eval.make_gsm8k --write     # fetch the official GSM8K test split -> 100 items
python -m eval.make_multistep --write # top ms_4/5/6step up to n=100 each
python -m eval.verify_stimuli         # re-derive every arithmetic answer from its text

python -m eval.pilot         # Step 1 gate: keep items models get right in prose
python -m eval.run_eval      # Steps 2-4: prose + json across 5 models -> results.json
python -m eval.make_heatmap  # Step 5: static heatmap.html (open in a browser)

# interactive explorer + live panel (Steps 6+8):
uvicorn web.server:app        # then open http://127.0.0.1:8000

# stretch:
python -m eval.run_eval --all        # add lexical + strict_schema conditions
python -m eval.ceiling_sweep --run   # depth sweep 8->40 steps -> ceiling.json (+ ceiling.html)
                                     # the explorer picks ceiling.json up automatically
```

Responses are cached under `cache/` keyed by (model, item, condition) **and validated
against the stored prompt**, so re-running is free, re-scoring after a scorer change
costs $0, and an item whose text changed under a stable id can never silently return a
stale answer. Cost is small: the full 450-item × 4-condition × 5-model run plus the
9,000-call depth sweep is **~$3.50 at current OpenRouter prices**, ~90% of it GPT-4o.

> **Need a key?** Get one at openrouter.ai/keys — a single key covers every model
> here (open + closed, across vendors). Some models have `:free` variants
> (rate-limited) if you want to run at $0: `python -m eval.list_models :free`.

---

## Layout

```
docs/METHODOLOGY.md        full methodology & code walkthrough (start here)
data/stimuli.json          450 items: 350 self-authored + 100 fetched GSM8K
eval/scorer.py             three-bucket scorer + compliance signals
eval/conditions.py         prose · json · lexical · strict_schema prompts
eval/harness.py            OpenRouter calls + prompt-checked response cache
eval/numwords.py           integer -> word spellings (acceptable_variants)
eval/make_gsm8k.py         fetch the official GSM8K test split -> the gsm8k slice
eval/make_multistep.py     top the ms_4/5/6step levels up to n=100
eval/verify_stimuli.py     re-derive every arithmetic answer from its question text
eval/list_models.py        list / verify OpenRouter model slugs
eval/pilot.py              Step 1: prose gate
eval/run_eval.py           Steps 2-4: run + score -> results.json
eval/make_heatmap.py       Step 5: static heatmap.html
eval/ceiling_sweep.py      ungated depth sweep -> ceiling.json + ceiling/level_*.json
tests/test_scorer.py       offline scorer tests (no key needed)
web/index.html             interactive explorer: heatmap -> drill-down, depth curve, live panel
web/server.py              tiny FastAPI backend (/results.json + /ceiling.json + /live)
```

---

## Citations

- **Tam et al., 2024** — *Let Me Speak Freely? A Study on the Impact of Format
  Restrictions on Performance of LLMs* — arXiv:2408.02442. *(The effect.)*
- **Fan, 2026** — *Capacity, Not Format: Rethinking Structured Reasoning Failures*
  — arXiv:2606.09410. *(The mechanism: capacity competition; reason-first-format-
  later recovers most loss.)*
- **Lee et al., 2026** — *The Format Tax* — arXiv:2604.03616.
- *One Token Away from Collapse* — arXiv:2604.13006. *(Lexical constraints.)*
- **Geng et al.** — *JSONSchemaBench* — arXiv:2501.10868. *(Cited to disclaim
  cross-library novelty.)*

---

*"The format tax is known. What this is: the instrument that lets you see it happen
on your own question, on today's models — and a scorer honest enough to tell 'the
model was wrong' apart from 'I couldn't parse it.'"*
