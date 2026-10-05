#!/usr/bin/env python3
"""Endpoint resolution shared by the generation scripts in this tree.

Every generation script speaks plain OpenAI-compatible `/v1/chat/completions`, so any
such server works: the internal NVIDIA inference API (the default, used for the frozen
anchors), a self-hosted vLLM / Modal deployment (see
`benchmarks/backend/modal_inference/`), or a provider API.

    EVAL_BASE_URL   full chat-completions URL, a `/v1` base, or a bare host.
                    default: the internal NVIDIA inference API.
    EVAL_API_KEY    credential; falls back to NVIDIA_API_KEY. Optional for a custom
                    endpoint that needs no auth, required for the default.

Examples:
    EVAL_BASE_URL=https://<user>--rtlrepair-qwen.modal.run  SLOT_MODEL=<served> ...
    EVAL_BASE_URL=http://localhost:8000/v1                  SLOT_MODEL=<served> ...

The credential is read from the environment only and is never written to a file, log,
prompt, or result artifact.

Not used by `benchmarks/backend/nvidia_responses.py`: the frozen preregistered primary
run goes through the NVIDIA *Responses* API, not chat/completions, and stays pinned.
"""
import os

DEFAULT_BASE_URL = "https://provider-a.invalid/v1"


def chat_url():
    """Normalize EVAL_BASE_URL into a chat/completions URL."""
    raw = (os.environ.get("EVAL_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    if not raw.endswith("/v1"):
        raw += "/v1"
    return raw + "/chat/completions"


def is_default_endpoint():
    return chat_url().startswith(DEFAULT_BASE_URL)


def auth_headers():
    """Request headers; Authorization is omitted when a custom endpoint needs no key."""
    headers = {"Content-Type": "application/json"}
    key = (os.environ.get("EVAL_API_KEY") or os.environ.get("NVIDIA_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    elif is_default_endpoint():
        raise SystemExit("credential required: set EVAL_API_KEY (or NVIDIA_API_KEY)")
    return headers
