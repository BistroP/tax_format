"""Tiny backend for the interactive front-end (Section 5.5).

Two jobs only, per the spec: serve the precomputed `results.json` (the spine),
and run one live question through the harness (the stretch). Keys stay server-side.

Run:  uvicorn web.server:app --reload      (from the repo root)
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from eval import config, harness, scorer

WEB = config.ROOT / "web"
LIVE_CONDITIONS = ["prose", "json"]  # the core contrast; fast enough for a live demo

app = FastAPI(title="Format-Tax Explorer")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/results.json")
def results():
    if config.RESULTS_PATH.exists():
        return FileResponse(config.RESULTS_PATH, media_type="application/json")
    return JSONResponse(
        {"error": "results.json not found — run `python -m eval.run_eval` first."},
        status_code=404,
    )


@app.get("/ceiling.json")
def ceiling():
    path = config.ROOT / "ceiling.json"
    if path.exists():
        return FileResponse(path, media_type="application/json")
    return JSONResponse(
        {"error": "ceiling.json not found — run `python -m eval.ceiling_sweep --run` first."},
        status_code=404,
    )


@app.get("/ceiling/level_{level}.json")
def ceiling_level(level: int):
    """Per-depth drill-down: the questions + every model's response at that level.
    `level: int` is the path guard — it makes a traversal path un-routable."""
    path = config.ROOT / "ceiling" / f"level_{level:02d}.json"
    if path.exists():
        return FileResponse(path, media_type="application/json")
    return JSONResponse({"error": f"no sweep data for {level} steps"}, status_code=404)


class LiveRequest(BaseModel):
    question: str
    expected: Optional[str] = None


def _run_one(model_cfg: dict, item: dict, condition: str, scored: bool) -> dict:
    is_json = condition in ("json", "strict_schema")
    try:
        rec = harness.query(model_cfg, item, condition,
                            use_cache=False, write_cache=False)
        text = rec["text"]
        return {
            "model_key": model_cfg["key"], "label": model_cfg["label"],
            "condition": condition, "text": text,
            "extracted": scorer.extract_answer(text) if is_json else None,
            "pure_json": scorer.is_pure_json(text) if is_json else None,
            "bucket": scorer.score(text, condition, item) if scored else None,
            "latency_s": rec.get("latency_s"), "error": None,
        }
    except Exception as e:  # a dead key / rate limit must not crash the page
        return {
            "model_key": model_cfg["key"], "label": model_cfg["label"],
            "condition": condition, "text": None, "extracted": None,
            "pure_json": None, "bucket": None, "latency_s": None,
            "error": f"{type(e).__name__}: {str(e)[:180]}",
        }


@app.post("/live")
def live(req: LiveRequest):
    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "Please enter a question."}, status_code=400)
    expected = (req.expected or "").strip()
    scored = bool(expected)

    item = {
        "id": "live_" + hashlib.sha1(question.encode()).hexdigest()[:8],
        "question": question,
        "canonical_answer": expected,
        "acceptable_variants": [],
        "category": "live",
        "banned_word": expected,
    }
    jobs = [(m, c) for m in config.MODELS for c in LIVE_CONDITIONS]
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        rows = list(ex.map(lambda mc: _run_one(mc[0], item, mc[1], scored), jobs))

    return {
        "question": question, "expected": expected, "scored": scored,
        "conditions": LIVE_CONDITIONS,
        "models": [{"key": m["key"], "label": m["label"]} for m in config.MODELS],
        "results": rows,
    }
