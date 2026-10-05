from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from backend.nvidia_responses import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    MissingNVIDIAAPIKey,
    NVIDIAHTTPError,
    NVIDIAResponsesClient,
    WireResponse,
)


class NVIDIAResponsesClientTests(unittest.TestCase):
    def test_missing_key_fails_before_transport(self):
        called = []

        def transport(request, timeout):
            called.append((request, timeout))
            raise AssertionError("transport should not be called")

        client = NVIDIAResponsesClient(transport=transport)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingNVIDIAAPIKey):
                client.create(prompt="p", system="s", idempotency_key="call-1")
        self.assertEqual(called, [])

    def test_frozen_responses_payload_and_env_only_auth(self):
        captured = {}

        def transport(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["idempotency"] = request.get_header("Idempotency-key")
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            body = {
                "id": "resp-1",
                "model": DEFAULT_MODEL,
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "module TopModule; endmodule",
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
            return WireResponse(200, json.dumps(body).encode(), {"x-request-id": "req-1"})

        client = NVIDIAResponsesClient(transport=transport)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-secret"}, clear=True):
            response = client.create(
                prompt="repair", system="strict", idempotency_key="call-fixed"
            )
        self.assertEqual(captured["url"], DEFAULT_BASE_URL + "/responses")
        self.assertEqual(captured["authorization"], "Bearer test-secret")
        self.assertEqual(captured["idempotency"], "call-fixed")
        self.assertEqual(captured["payload"]["model"], DEFAULT_MODEL)
        self.assertEqual(captured["payload"]["reasoning"], {"effort": "high"})
        self.assertEqual(
            captured["payload"]["max_output_tokens"], DEFAULT_MAX_OUTPUT_TOKENS
        )
        self.assertEqual(response.output_text, "module TopModule; endmodule")
        self.assertEqual(response.request_id, "req-1")
        self.assertNotIn("test-secret", json.dumps(response.storage_record()))
        self.assertNotIn("authorization", json.dumps(response.storage_record()).lower())

    def test_echoed_secret_and_header_fields_are_redacted(self):
        def transport(request, timeout):
            body = {
                "status": "completed",
                "output_text": "secret-value",
                "headers": {"Authorization": "Bearer secret-value"},
                "debug": {"api_key": "secret-value"},
            }
            return WireResponse(200, json.dumps(body).encode())

        client = NVIDIAResponsesClient(transport=transport)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "secret-value"}, clear=True):
            response = client.create(prompt="p", system="s", idempotency_key="i")
        serialized = json.dumps(response.storage_record())
        self.assertNotIn("secret-value", serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_refusal_and_incomplete_are_http_200_model_outcomes(self):
        responses = [
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "cannot comply"}],
                    }
                ],
            },
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": "module TopModule;",
            },
        ]

        def transport(request, timeout):
            return WireResponse(200, json.dumps(responses.pop(0)).encode())

        client = NVIDIAResponsesClient(transport=transport)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            refusal = client.create(prompt="p", system="s", idempotency_key="a")
            incomplete = client.create(prompt="p", system="s", idempotency_key="b")
        self.assertTrue(refusal.is_refusal)
        self.assertFalse(refusal.is_incomplete)
        self.assertTrue(incomplete.is_incomplete)
        self.assertEqual(incomplete.incomplete_reason, "max_output_tokens")

    def test_non_200_status_is_transport_error_without_body_or_headers(self):
        def transport(request, timeout):
            return WireResponse(
                429,
                b'{"debug":"secret"}',
                {"Retry-After": "3", "Authorization": "Bearer secret"},
            )

        client = NVIDIAResponsesClient(transport=transport)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "secret"}, clear=True):
            with self.assertRaises(NVIDIAHTTPError) as raised:
                client.create(prompt="p", system="s", idempotency_key="i")
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(raised.exception.retry_after_seconds, 3.0)
        self.assertNotIn("secret", json.dumps(raised.exception.safe_record()))

    def test_invalid_http_200_json_is_returned_as_protocol_outcome(self):
        client = NVIDIAResponsesClient(
            transport=lambda request, timeout: WireResponse(200, b"not-json")
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            response = client.create(prompt="p", system="s", idempotency_key="i")
        self.assertEqual(response.status, "protocol_error")
        self.assertIsNotNone(response.protocol_error)

    def test_served_model_mismatch_is_terminal_protocol_outcome(self):
        body = {
            "model": "some/other-model",
            "status": "completed",
            "output_text": "module TopModule; endmodule",
        }
        client = NVIDIAResponsesClient(
            transport=lambda request, timeout: WireResponse(
                200, json.dumps(body).encode()
            )
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            response = client.create(prompt="p", system="s", idempotency_key="i")
        self.assertEqual(response.model, "some/other-model")
        self.assertIsNotNone(response.protocol_error)


if __name__ == "__main__":
    unittest.main()
