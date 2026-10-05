"""Strict NVIDIA-hosted OpenAI Responses client for VeriCodeGen experiments.

This module is intentionally separate from :mod:`backend.llm_client`.  The
research protocol must never fall back to a stub or to another model, and the
credential must only come from ``NVIDIA_API_KEY`` at call time.

Only one wire request is made by :meth:`NVIDIAResponsesClient.create`.  Retry
policy and its accounting live in ``benchmarks/vericodegen_eval.py`` so every
transport attempt can be written to the append-only experiment ledger before
it is sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import socket
import time
from typing import Any, Callable, Mapping, Optional
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "https://provider-a.invalid/v1"
DEFAULT_MODEL = "azure/openai/gpt-5.5"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_REASONING_EFFORT = "high"
API_KEY_ENV = "NVIDIA_API_KEY"

_RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_SECRET_FIELD_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "api_key",
        "apikey",
        "x-api-key",
        "headers",
    }
)


class NVIDIAResponsesError(RuntimeError):
    """Base class whose public attributes are safe to persist."""

    retryable = False
    status_code: Optional[int] = None
    retry_after_seconds: Optional[float] = None
    request_id: Optional[str] = None

    def safe_record(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": str(self),
            "retryable": bool(self.retryable),
            "status_code": self.status_code,
            "retry_after_seconds": self.retry_after_seconds,
            "request_id": self.request_id,
        }


class MissingNVIDIAAPIKey(NVIDIAResponsesError):
    """Raised before any request if ``NVIDIA_API_KEY`` is absent."""


class NVIDIAHTTPError(NVIDIAResponsesError):
    """Non-200 HTTP response, deliberately excluding response headers/body."""

    def __init__(
        self,
        status_code: int,
        *,
        retry_after_seconds: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(f"NVIDIA Responses API returned HTTP {status_code}")
        self.status_code = int(status_code)
        self.retryable = self.status_code in _RETRYABLE_HTTP
        self.retry_after_seconds = retry_after_seconds
        self.request_id = request_id


class NVIDIATransportError(NVIDIAResponsesError):
    """Network/timeout failure before a usable HTTP response."""

    retryable = True

    def __init__(self, category: str) -> None:
        # Keep this message categorical.  Arbitrary exception strings can
        # contain URLs or diagnostic request dumps and are not persisted.
        super().__init__(f"NVIDIA Responses transport failure ({category})")
        self.category = category


@dataclass(frozen=True)
class WireResponse:
    """Small transport-neutral HTTP response used by tests and urllib."""

    status_code: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NVIDIAResponse:
    """A sanitized HTTP-200 Responses result.

    ``protocol_error``, refusal and incomplete status are model outcomes, not
    transport exceptions.  The caller must record them in the research
    denominator.
    """

    raw_response: Any
    output_text: str
    response_id: Optional[str]
    model: str
    status: str
    usage: Mapping[str, Any]
    refusal: Optional[str]
    incomplete_reason: Optional[str]
    protocol_error: Optional[str]
    request_id: Optional[str]
    elapsed_ms: float

    @property
    def is_refusal(self) -> bool:
        return bool(self.refusal)

    @property
    def is_incomplete(self) -> bool:
        return self.status != "completed" or bool(self.incomplete_reason)

    def storage_record(self) -> dict[str, Any]:
        """Return the full response record; it never includes request headers."""

        return {
            "raw_response": self.raw_response,
            "output_text": self.output_text,
            "response_id": self.response_id,
            "served_model": self.model,
            "response_status": self.status,
            "usage": dict(self.usage),
            "refusal": self.refusal,
            "incomplete_reason": self.incomplete_reason,
            "protocol_error": self.protocol_error,
            "request_id": self.request_id,
            "elapsed_ms": self.elapsed_ms,
        }


Transport = Callable[[urllib.request.Request, float], WireResponse]


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return None


def _retry_after(headers: Mapping[str, str]) -> Optional[float]:
    value = _header(headers, "retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _sanitize_for_storage(value: Any, secret: str) -> Any:
    """Recursively redact credential-shaped fields and an echoed key."""

    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).strip().lower() in _SECRET_FIELD_NAMES:
                clean[str(key)] = "[REDACTED]"
            else:
                clean[str(key)] = _sanitize_for_storage(item, secret)
        return clean
    if isinstance(value, list):
        return [_sanitize_for_storage(item, secret) for item in value]
    # Avoid corrupting ordinary response text when a unit-test/development key
    # is a one-character placeholder.  Real NVIDIA credentials are much
    # longer; credential-shaped mapping fields are redacted regardless.
    if isinstance(value, str) and len(secret) >= 8:
        return value.replace(secret, "[REDACTED]")
    return value


def _extract_output(payload: Mapping[str, Any]) -> tuple[str, Optional[str]]:
    top = payload.get("output_text")
    texts: list[str] = []
    refusals: list[str] = []
    top_refusal = payload.get("refusal")
    if isinstance(top_refusal, str):
        refusals.append(top_refusal)
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                kind = str(part.get("type") or "")
                text = part.get("text")
                if kind == "output_text" and isinstance(text, str):
                    texts.append(text)
                if kind == "refusal":
                    refusal = part.get("refusal")
                    if isinstance(refusal, str):
                        refusals.append(refusal)
                    elif isinstance(text, str):
                        refusals.append(text)
    if not texts and isinstance(top, str):
        texts.append(top)
    return "\n".join(texts), ("\n".join(refusals) or None)


def _parse_http_200(
    wire: WireResponse,
    *,
    expected_model: str,
    secret: str,
    elapsed_ms: float,
) -> NVIDIAResponse:
    request_id = _header(wire.headers, "x-request-id") or _header(
        wire.headers, "request-id"
    )
    decoded = wire.body.decode("utf-8", errors="replace")
    try:
        parsed: Any = json.loads(decoded)
    except json.JSONDecodeError:
        return NVIDIAResponse(
            raw_response=_sanitize_for_storage(decoded, secret),
            output_text="",
            response_id=None,
            model=expected_model,
            status="protocol_error",
            usage={},
            refusal=None,
            incomplete_reason=None,
            protocol_error="HTTP 200 body was not valid JSON",
            request_id=request_id,
            elapsed_ms=elapsed_ms,
        )

    clean = _sanitize_for_storage(parsed, secret)
    if not isinstance(clean, Mapping):
        return NVIDIAResponse(
            raw_response=clean,
            output_text="",
            response_id=None,
            model=expected_model,
            status="protocol_error",
            usage={},
            refusal=None,
            incomplete_reason=None,
            protocol_error="HTTP 200 JSON body was not an object",
            request_id=request_id,
            elapsed_ms=elapsed_ms,
        )

    text, refusal = _extract_output(clean)
    status = str(clean.get("status") or "completed")
    served_model = str(clean.get("model") or expected_model)
    protocol_error = None
    if served_model != expected_model:
        # A successful gateway response from another model is still an HTTP-200
        # model outcome, not a retryable transport error.  Mark it terminal so
        # the preregistered no-fallback contract cannot silently accept it.
        protocol_error = "HTTP 200 response reported an unexpected served model"
    incomplete = clean.get("incomplete_details")
    incomplete_reason: Optional[str] = None
    if isinstance(incomplete, Mapping) and incomplete.get("reason") is not None:
        incomplete_reason = str(incomplete.get("reason"))
    elif status != "completed":
        incomplete_reason = status
    usage = clean.get("usage")
    return NVIDIAResponse(
        raw_response=clean,
        output_text=text,
        response_id=(str(clean["id"]) if clean.get("id") is not None else None),
        model=served_model,
        status=status,
        usage=(dict(usage) if isinstance(usage, Mapping) else {}),
        refusal=refusal,
        incomplete_reason=incomplete_reason,
        protocol_error=protocol_error,
        request_id=request_id,
        elapsed_ms=elapsed_ms,
    )


def _urllib_transport(request: urllib.request.Request, timeout: float) -> WireResponse:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers = {str(k): str(v) for k, v in response.headers.items()}
            return WireResponse(
                status_code=int(response.status),
                body=response.read(),
                headers=headers,
            )
    except urllib.error.HTTPError as exc:
        # Do not read or retain the error body: it is not needed for retry
        # accounting and occasionally contains request diagnostics.
        headers = {str(k): str(v) for k, v in (exc.headers or {}).items()}
        error = NVIDIAHTTPError(
            int(exc.code),
            retry_after_seconds=_retry_after(headers),
            request_id=_header(headers, "x-request-id"),
        )
        exc.close()
        raise error from None
    except (urllib.error.URLError, TimeoutError, socket.timeout):
        raise NVIDIATransportError("network_or_timeout") from None
    except OSError:
        raise NVIDIATransportError("operating_system") from None


class NVIDIAResponsesClient:
    """Minimal strict client for NVIDIA's OpenAI-compatible Responses route."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        timeout_seconds: float = 180.0,
        transport: Optional[Transport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport or _urllib_transport

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/responses"

    def request_metadata(self) -> dict[str, Any]:
        """Sanitized, stable request metadata suitable for the ledger."""

        return {
            "provider": "nvidia",
            "wire_api": "responses",
            "endpoint": self.endpoint,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "fallback": False,
        }

    def create(
        self,
        *,
        prompt: str,
        system: str,
        idempotency_key: str,
    ) -> NVIDIAResponse:
        """Make exactly one Responses request and return an HTTP-200 result."""

        api_key = os.environ.get(API_KEY_ENV, "").strip()
        if not api_key:
            raise MissingNVIDIAAPIKey(
                f"{API_KEY_ENV} is required for a paid NVIDIA request"
            )

        payload = {
            "model": self.model,
            "instructions": system,
            "input": prompt,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        started = time.monotonic()
        wire = self._transport(request, self.timeout_seconds)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if int(wire.status_code) != 200:
            raise NVIDIAHTTPError(
                int(wire.status_code),
                retry_after_seconds=_retry_after(wire.headers),
                request_id=_header(wire.headers, "x-request-id"),
            )
        return _parse_http_200(
            wire,
            expected_model=self.model,
            secret=api_key,
            elapsed_ms=elapsed_ms,
        )


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_REASONING_EFFORT",
    "MissingNVIDIAAPIKey",
    "NVIDIAHTTPError",
    "NVIDIAResponse",
    "NVIDIAResponsesClient",
    "NVIDIAResponsesError",
    "NVIDIATransportError",
    "WireResponse",
]
