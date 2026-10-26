
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main


class UnexpectedShapeResponse:
    status_code = 200

    def __init__(self):
        # Valid JSON, but not the object expected from the provider.
        self.body = json.dumps(
            ["unexpected", "provider", "response"]
        ).encode("utf-8")
        self.headers = {
            "Content-Length": str(len(self.body)),
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def aiter_bytes(self):
        yield self.body

    def raise_for_status(self):
        return None


class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        # Accept the same constructor arguments used by httpx.AsyncClient.
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def stream(self, method, url, headers=None, json=None):
        return UnexpectedShapeResponse()


class OpenAIResponseShapeTests(unittest.TestCase):
    def setUp(self):
        self.original_values = {
            "GREYFLOW_API_KEY": main.GREYFLOW_API_KEY,
            "OPENAI_API_KEY": main.OPENAI_API_KEY,
            "IMAGE_MODEL": main.IMAGE_MODEL,
        }

        main.GREYFLOW_API_KEY = "test-greyflow-key"
        main.OPENAI_API_KEY = "test-openai-key"
        main.IMAGE_MODEL = "test-image-model"

        self.client_patch = patch.object(
            main.httpx,
            "AsyncClient",
            FakeAsyncClient,
        )
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

        self.client = TestClient(
            main.app,
            raise_server_exceptions=False,
        )

    def tearDown(self):
        for name, value in self.original_values.items():
            setattr(main, name, value)

    def test_unexpected_top_level_json_returns_safe_502(self):
        response = self.client.post(
            "/v1/images/generations",
            headers={
                "X-GreyFlow-API-Key": "test-greyflow-key",
            },
            json={"prompt": "A small red house"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("test-openai-key", response.text)
        self.assertNotIn("AttributeError", response.text)
        self.assertNotIn("TypeError", response.text)


if __name__ == "__main__":
    unittest.main()
