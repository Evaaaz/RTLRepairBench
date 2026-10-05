"""Thin client for an OpenAI-compatible vLLM endpoint, with offline fallback.

`generate(...)` hits the chat.completions API at ``RTLREPAIR_LLM_URL``
(default ``http://localhost:8000/v1``). On ANY failure -- no server, network
error, bad response, missing openai package -- it falls back to
``backend.serve_stub.canned_rtl(prompt)`` so the pipeline always returns
usable text.

For callers that need to *know* whether the result came from the live model
or from the stub (the benchmark harness), three additions over the original
API:

* ``last_call_status()`` -- inspect the most recent generate() call: which
  model was hit, latency, whether the stub took over, and the error if any.
* ``generate_strict(...)`` / ``strict=True`` -- raise instead of falling back,
  so a 401 or a network failure cannot silently masquerade as canned output.
* ``health_check()`` -- one-shot ``GET /models`` ping returning ``{ok, models}``.

The app lane's streaming path (``stream_chat``) is not carried here: nothing in
the eval harness consumes token deltas. It lives in the root ``backend/`` copy.

IMPORTANT: ``openai`` is imported lazily INSIDE the call sites so that
``import backend.llm_client`` works with only the stdlib installed.
"""

from __future__ import annotations

import json as _json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Default OpenAI-compatible vLLM endpoint.
DEFAULT_LLM_URL = "http://localhost:8000/v1"
# Placeholder key; vLLM ignores it on open endpoints, but the openai client
# requires a non-empty value. Authenticated endpoints set RTLREPAIR_LLM_API_KEY.
DEFAULT_API_KEY = "EMPTY"

# ---------------------------------------------------------------------------
# Last-call status: lets the UI render an honest "live tuned" / "stub" badge
# instead of letting a 401 look like a working model returning canned output.
# ---------------------------------------------------------------------------
_STATUS_DEFAULTS: Dict[str, Any] = {
    "ok": None,             # bool | None: did the live LLM call succeed?
    "served_model": None,   # str | None: which model id was actually requested
    "elapsed_ms": None,     # float | None
    "error": None,          # str | None
    "stub_fallback": None,  # bool | None: did serve_stub take over?
    "finish_reasons": None,  # list[str] | None: per-choice OpenAI finish_reason
    "usage": None,          # dict | None: {prompt_tokens, completion_tokens, total_tokens}
    "ts": 0.0,              # float: time.time() of last update
}

_status_lock = threading.Lock()
# served_model -> sampling params the endpoint rejected, so a retry is only paid once
_unsupported_params: Dict[str, set] = {}
_last_call_status: Dict[str, Any] = dict(_STATUS_DEFAULTS)

# Per-thread status. A concurrent eval runs many generate() calls at once, and a
# process-global "last call" would let one worker read another's finish_reason /
# elapsed_ms. Readers get their OWN thread's last call when there is one, and
# fall back to the global for single-threaded callers (app.py's status badge).
_status_local = threading.local()


def _set_status(**kwargs) -> None:
    prev = getattr(_status_local, "status", None)
    entry = dict(prev) if prev is not None else dict(_STATUS_DEFAULTS)
    entry.update(kwargs)
    entry["ts"] = time.time()
    _status_local.status = entry
    with _status_lock:
        _last_call_status.update(entry)


def last_call_status() -> Dict[str, Any]:
    """Return a snapshot of the most-recent generate() call.

    Keys: ok, served_model, elapsed_ms, error, stub_fallback, finish_reasons,
    usage, ts. Values may be ``None`` before the first call.

    Every field is set on every path (``None`` where it does not apply) because
    ``_set_status`` merges into the thread's previous entry — an omitted key
    would silently report the *previous* call's value.

    Returns the CALLING THREAD's most recent call when that thread has made one,
    so concurrent workers never read each other's status; otherwise the global.
    """
    local = getattr(_status_local, "status", None)
    if local is not None:
        return dict(local)
    with _status_lock:
        return dict(_last_call_status)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _served_model(model: str) -> str:
    """Map an app-level model id to the id the vLLM server actually serves.

    "tuned+repair" is not a separately served weight: it is the "tuned" LoRA
    adapter plus the harness's repair loop (``benchmarks/repair_agent.py``).
    For the raw generate() call both resolve to the same served adapter, so it
    maps to "tuned".
    Every other id (``base``, ``tuned``, ``tunedv6``, ...) passes through.
    """
    return "tuned" if model == "tuned+repair" else model


def _fallback(prompt: str, n: int) -> List[str]:
    """Offline path: return canned RTL from serve_stub, ``n`` copies."""
    # Imported lazily to keep this module stdlib-only at import time.
    from backend import serve_stub

    text = serve_stub.canned_rtl(prompt)
    return [text for _ in range(max(1, n))]


def _build_messages(prompt: str, system: Optional[str] = None) -> List[Dict[str, str]]:
    sys_text = system or (
        "You are an expert hardware design assistant. "
        "Respond with synthesizable SystemVerilog only when asked for RTL."
    )
    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": prompt},
    ]


def _endpoint() -> str:
    return os.environ.get("RTLREPAIR_LLM_URL", DEFAULT_LLM_URL)


def _api_key() -> str:
    return os.environ.get("RTLREPAIR_LLM_API_KEY", DEFAULT_API_KEY)


def _client_or_error():
    """Return (OpenAI_client_or_None, error_text_or_None)."""
    base_url = _endpoint()
    api_key = _api_key()
    try:
        from openai import OpenAI

        return OpenAI(base_url=base_url, api_key=api_key), None
    except Exception as exc:  # noqa: BLE001 - any import/init failure is a usable error
        return None, "openai client unavailable: {0}: {1}".format(
            type(exc).__name__, exc
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate(
    prompt: str,
    model: str = "base",
    n: int = 1,
    temp: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    *,
    system: Optional[str] = None,
    strict: bool = False,
) -> List[str]:
    """Generate ``n`` completions for ``prompt`` from the served model.

    Args:
        prompt: user message text.
        model: app-level model id (``base``, ``tuned``, ``tunedv6``, ...).
        n, temp, top_p, max_tokens: standard OpenAI sampling knobs.
        system: optional system prompt override (defaults to the hardware
            assistant prompt).
        strict: if True, raise on any failure instead of silently falling
            back to ``serve_stub``. Use this for paths where the IDE/UI must
            distinguish a live model from a stub (repair, chat, etc.).

    Returns:
        A list of ``n`` strings on success. On failure: either raises
        (``strict=True``) or returns a canned-RTL fallback (``strict=False``).
        Either way ``last_call_status()`` records what happened.
    """
    served_model = _served_model(model)
    t0 = time.monotonic()
    client, err = _client_or_error()

    if client is None:
        _set_status(
            ok=False,
            served_model=served_model,
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            error=err,
            stub_fallback=not strict,
            finish_reasons=None,
            usage=None,
        )
        if strict:
            raise RuntimeError(err)
        return _fallback(prompt, n)

    def _create(**overrides):
        kw = dict(
            model=served_model,
            n=max(1, n),
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            messages=_build_messages(prompt, system=system),
        )
        kw.update(overrides)
        return client.chat.completions.create(**{k: v for k, v in kw.items() if v is not None})

    try:
        try:
            resp = _create()
        except Exception as exc:  # noqa: BLE001
            # Reasoning models reject some standard sampling knobs ("Unsupported
            # parameter: 'top_p' is not supported with this model"). Retry without the
            # named parameter instead of letting the caller fall back to the rule-based
            # path, which would silently record the model as recovering nothing.
            # The parameter name arrives inside a nested, escaped JSON payload, so the
            # quotes around it may be backslash-escaped.
            m = re.search(r"Unsupported parameter: \\?'([a-z_]+)\\?'", str(exc))
            if not m:
                raise
            dropped = m.group(1)
            resp = _create(**{dropped: None})
            _unsupported_params.setdefault(served_model, set()).add(dropped)
        texts = [(c.message.content or "") for c in resp.choices]
        finishes = [c.finish_reason for c in resp.choices]
        # Token counts are the only cost signal the ledger can carry, and they
        # exist only on the response object — capture them before it goes away.
        usage = getattr(resp, "usage", None)
        usage = ({"prompt_tokens": getattr(usage, "prompt_tokens", None),
                  "completion_tokens": getattr(usage, "completion_tokens", None),
                  "total_tokens": getattr(usage, "total_tokens", None)}
                 if usage is not None else None)
        if not texts or all(not t.strip() for t in texts):
            msg = "empty response from {0}".format(served_model)
            _set_status(
                ok=False,
                served_model=served_model,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                error=msg,
                stub_fallback=not strict,
                finish_reasons=finishes,
                usage=usage,
            )
            if strict:
                raise RuntimeError(msg)
            return _fallback(prompt, n)
        _set_status(
            ok=True,
            served_model=served_model,
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            error=None,
            stub_fallback=False,
            finish_reasons=finishes,
            usage=usage,
        )
        return texts
    except Exception as exc:  # network, auth, server-side, etc.
        msg = "{0}: {1}".format(type(exc).__name__, exc)
        _set_status(
            ok=False,
            served_model=served_model,
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            error=msg,
            stub_fallback=not strict,
            finish_reasons=None,
            usage=None,
        )
        if strict:
            raise
        return _fallback(prompt, n)


def generate_strict(
    prompt: str,
    model: str = "tuned",
    n: int = 1,
    temp: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    *,
    system: Optional[str] = None,
) -> List[str]:
    """Like ``generate`` but raises on any failure (no stub fallback)."""
    return generate(
        prompt,
        model=model,
        n=n,
        temp=temp,
        top_p=top_p,
        max_tokens=max_tokens,
        system=system,
        strict=True,
    )


def health_check(timeout_s: float = 5.0) -> Dict[str, Any]:
    """One-shot ``GET /models`` against the configured endpoint.

    Returns a dict with::

        {"ok": bool,
         "base_url": str,
         "models": list[str],     # served model ids, empty on failure
         "error": str | None,
         "auth": "bearer" | "none"}

    Implemented with stdlib ``urllib`` so the import does not pull ``openai``.
    """
    base_url = _endpoint()
    api_key = _api_key()
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": "Bearer {0}".format(api_key)} if api_key else {}
    auth = "bearer" if api_key and api_key != DEFAULT_API_KEY else "none"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            payload = _json.loads(r.read().decode("utf-8", errors="replace"))
        models = [m.get("id") for m in payload.get("data", []) if m.get("id")]
        return {
            "ok": True,
            "base_url": base_url,
            "models": models,
            "error": None,
            "auth": auth,
        }
    except urllib.error.HTTPError as exc:  # 401, 404, 500, ...
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        return {
            "ok": False,
            "base_url": base_url,
            "models": [],
            "error": "HTTP {0}: {1}".format(exc.code, body or exc.reason),
            "auth": auth,
        }
    except Exception as exc:  # noqa: BLE001 - urllib/network/json failures
        return {
            "ok": False,
            "base_url": base_url,
            "models": [],
            "error": "{0}: {1}".format(type(exc).__name__, exc),
            "auth": auth,
        }
