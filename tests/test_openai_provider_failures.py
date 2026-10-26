
import base64
import io
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from app import main


def png_header_base64(width=1024, height=1024):
    """Build a complete, valid PNG fixture in memory."""
    buffer = io.BytesIO()
    image = Image.new("RGB", (width, height), color=(20, 40, 60))
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=None):
        self.payload = payload if payload is not None else {}
        self.status_code = status_code
        self.json_error = json_error
        self.request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/images/generations",
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(
                self.status_code,
                request=self.request,
            )
            raise httpx.HTTPStatusError(
                "Mock provider error",
                request=self.request,
                response=response,
            )

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeAsyncClient:
    next_payload = {}
    next_status_code = 200
    next_exception = None
    next_json_error = None
    calls = []

    def __init__(self, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append({
            "url": url,
            "headers": headers or {},
            "json": json,
        })

        if self.next_exception is not None:
            raise self.next_exception

        return FakeResponse(
            payload=self.next_payload,
            status_code=self.next_status_code,
            json_error=self.next_json_error,
        )


class OpenAIProviderFailureTests(unittest.TestCase):
    def setUp(self):
        self.original_values = {
            "GREYFLOW_API_KEY": main.GREYFLOW_API_KEY,
            "OPENAI_API_KEY": main.OPENAI_API_KEY,
            "IMAGE_MODEL": main.IMAGE_MODEL,
        }

        main.GREYFLOW_API_KEY = "test-greyflow-key"
        main.OPENAI_API_KEY = "test-openai-key"
        main.IMAGE_MODEL = "test-image-model"

        FakeAsyncClient.next_payload = {}
        FakeAsyncClient.next_status_code = 200
        FakeAsyncClient.next_exception = None
        FakeAsyncClient.next_json_error = None
        FakeAsyncClient.calls = []

        self.client_patch = patch.object(
            main.httpx,
            "AsyncClient",
            FakeAsyncClient,
        )
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

        self.client = TestClient(main.app)

    def tearDown(self):
        for name, value in self.original_values.items():
            setattr(main, name, value)

    def headers(self):
        return {
            "X-GreyFlow-API-Key": "test-greyflow-key",
        }

    def generate(self):
        return self.client.post(
            "/v1/images/generations",
            headers=self.headers(),
            json={"prompt": "A small red house"},
        )

    def test_missing_greyflow_api_key_returns_401(self):
        response = self.client.post(
            "/v1/images/generations",
            json={"prompt": "A small red house"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(FakeAsyncClient.calls, [])

    def test_missing_openai_configuration_returns_503(self):
        main.OPENAI_API_KEY = ""

        response = self.generate()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(FakeAsyncClient.calls, [])

    def test_provider_rate_limit_returns_429(self):
        FakeAsyncClient.next_status_code = 429

        response = self.generate()

        self.assertEqual(response.status_code, 429)
        self.assertIn("rate limit", response.json()["detail"].lower())
        self.assertNotIn("test-openai-key", response.text)

    def test_provider_auth_failure_returns_safe_502(self):
        for status_code in (401, 403):
            with self.subTest(status_code=status_code):
                FakeAsyncClient.next_status_code = status_code

                response = self.generate()

                self.assertEqual(response.status_code, 502)
                self.assertNotIn("test-openai-key", response.text)
                self.assertNotIn("Bearer", response.text)

    def test_provider_timeout_returns_504(self):
        FakeAsyncClient.next_exception = httpx.ReadTimeout(
            "Mock provider timeout"
        )

        response = self.generate()

        self.assertEqual(response.status_code, 504)
        self.assertNotIn("Mock provider timeout", response.text)
        self.assertNotIn("test-openai-key", response.text)

    def test_connection_failure_returns_safe_502(self):
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/images/generations",
        )
        FakeAsyncClient.next_exception = httpx.ConnectError(
            "Mock connection failure",
            request=request,
        )

        response = self.generate()

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("Mock connection failure", response.text)
        self.assertNotIn("test-openai-key", response.text)

    def test_malformed_provider_json_returns_safe_502(self):
        FakeAsyncClient.next_json_error = ValueError(
            "Malformed provider JSON"
        )

        response = self.generate()

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("Malformed provider JSON", response.text)

    def test_empty_image_response_returns_502(self):
        FakeAsyncClient.next_payload = {"data": []}

        response = self.generate()

        self.assertEqual(response.status_code, 502)

    def test_unusable_image_data_returns_502(self):
        FakeAsyncClient.next_payload = {
            "data": [{"b64_json": "not-valid-base64!"}],
        }

        response = self.generate()

        self.assertEqual(response.status_code, 502)

    def test_successful_provider_response_uses_server_credentials(self):
        FakeAsyncClient.next_payload = {
            "data": [{"b64_json": png_header_base64()}],
        }

        response = self.generate()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["status"],
            "completed",
        )
        self.assertEqual(len(FakeAsyncClient.calls), 1)

        call = FakeAsyncClient.calls[0]
        self.assertEqual(
            call["headers"]["Authorization"],
            "Bearer test-openai-key",
        )
        self.assertEqual(
            call["json"]["model"],
            "test-image-model",
        )


if __name__ == "__main__":
    unittest.main()
