"""Model harness (Section 5.4): query models via OpenRouter (one key, many
providers) under each condition, caching raw responses to disk keyed by
(model, item_id, condition) so front-end/scorer changes never re-hit the API.

OpenRouter speaks the OpenAI Chat Completions API, so we use the `openai` SDK
pointed at OpenRouter's base URL. The key is read from `.env` (server-side only
— never in client JS). We read the dotenv file directly so an ambient env var
can't silently override it.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from . import config
from .conditions import build_prompt

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_client = None


def _read_dotenv() -> dict:
    env: dict[str, str] = {}
    path = config.ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get_client():
    global _client
    if _client is None:
        import os

        from openai import OpenAI

        env = _read_dotenv()
        api_key = env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. Copy .env.example to .env and add "
                "your key (get one at https://openrouter.ai/keys)."
            )
        base_url = env.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL
        _client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"X-Title": "Format-Tax Explorer"},
        )
    return _client


def _cache_path(model_key: str, item_id: str, condition: str):
    config.CACHE.mkdir(exist_ok=True)
    return config.CACHE / f"{model_key}__{item_id}__{condition}.json"


def _cached(model_key: str, item_id: str, condition: str, prompt: str) -> Optional[dict]:
    """Return the cached record only if *this exact prompt* produced it.

    The file is keyed by (model, item, condition), but an item's text can change
    under a stable id — bumping the sweep's PER_LEVEL reshuffles the generator, so
    `deep12_00` keeps its id and gets a new question. Comparing the stored prompt
    turns a silent stale hit into a plain miss.
    """
    path = _cache_path(model_key, item_id, condition)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return rec if rec.get("prompt") == prompt else None


def _is_transient_provider_error(msg: str) -> bool:
    """OpenRouter often surfaces an upstream 429 / provider hiccup as a 400 after
    it fails over between providers. Those are worth retrying; a genuinely
    malformed request is not."""
    m = msg.lower()
    return any(s in m for s in (
        "provider returned error", "rate-limit", "rate limit",
        "temporarily", "overloaded", "timeout", "no instances available",
    ))


def _create_with_retry(client, kwargs, retries: int = 4):
    import openai

    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APIConnectionError,
                openai.InternalServerError) as err:
            last = err
        except openai.BadRequestError as err:
            if not _is_transient_provider_error(str(err)):
                raise  # real bad request — surface it, don't loop
            last = err
        if attempt == retries - 1:
            raise last
        time.sleep(1.6 ** attempt + 0.5)


def query(model_cfg: dict, item: dict, condition: str,
          use_cache: bool = True, write_cache: bool = True,
          verbose: bool = False, max_tokens: int = None) -> dict:
    """Return a cached-or-fresh raw response record for (model, item, condition).

    `write_cache=False` (used by the live panel) keeps ad-hoc questions out of the
    benchmark cache. `max_tokens` overrides the default (e.g. the ceiling sweep
    needs a big budget so long chain-of-thought isn't truncated)."""
    prompt = build_prompt(condition, item)
    if use_cache:
        hit = _cached(model_cfg["key"], item["id"], condition, prompt)
        if hit is not None:
            return hit

    kwargs = {
        "model": model_cfg["model"],
        "max_tokens": max_tokens or config.MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    temp: Optional[float] = model_cfg.get("temperature")
    if temp is not None:
        kwargs["temperature"] = temp
    extra = model_cfg.get("extra")  # e.g. Gemini: disable internal reasoning
    if extra:
        kwargs["extra_body"] = extra
    # NB: no reasoning param for the rest — instruct-only models, on purpose (config.REASONING).

    t0 = time.time()
    resp = _create_with_retry(get_client(), kwargs)
    choice = resp.choices[0]
    record = {
        "model": model_cfg["model"],
        "model_key": model_cfg["key"],
        "item_id": item["id"],
        "condition": condition,
        "prompt": prompt,
        "text": choice.message.content or "",
        "stop_reason": choice.finish_reason,
        "latency_s": round(time.time() - t0, 2),
    }
    if write_cache:
        _cache_path(model_cfg["key"], item["id"], condition).write_text(
            json.dumps(record, indent=2, ensure_ascii=False))
    if verbose:
        print(f"    [{model_cfg['key']:8}] {item['id']:8} {condition:12} "
              f"({record['latency_s']}s)")
    return record


def prefetch(model_cfgs, items, conditions, max_workers: int = 8,
             verbose: bool = True, max_tokens: int = None) -> int:
    """Warm the on-disk cache in parallel so the sequential scoring pass is all
    cache hits. Only fetches (model, item, condition) combos not already cached."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from functools import partial

    get_client()  # initialise the singleton before spawning threads
    todo = [(m, it, c) for m in model_cfgs for c in conditions for it in items
            if _cached(m["key"], it["id"], c, build_prompt(c, it)) is None]
    if not todo:
        return 0
    done = failed = 0
    _q = partial(query, max_tokens=max_tokens)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_q, m, it, c): (m["key"], it["id"], c)
                for (m, it, c) in todo}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                failed += 1
                mk, iid, c = futs[f]
                print(f"  ! prefetch {mk}/{iid}/{c}: {type(e).__name__}: {str(e)[:90]}")
            done += 1
            if verbose and done % 50 == 0:
                print(f"    prefetched {done}/{len(todo)}")
    if verbose:
        print(f"    prefetched {len(todo)} new calls ({failed} failed)")
    return done
