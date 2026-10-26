
import base64
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        # Valid Base64, but the decoded bytes are not an image.
        encoded = base64.b64encode(b"this is not an image").decode("ascii")
        return {"data": [{"b64_json": encoded}]}


class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, headers=None, json=None):
        return FakeResponse()


class ImagePayloadValidationTests(unittest.TestCase):
    def setUp(self):
        self.original_values = {
            "GREYFLOW_API_KEY": main.GREYFLOW_API_KEY,
            "OPENAI_API_KEY": main.OPENAI_API_KEY,
        }
        main.GREYFLOW_API_KEY = "test-greyflow-key"
        main.OPENAI_API_KEY = "test-openai-key"

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

    def test_valid_base64_that_is_not_an_image_is_rejected(self):
        response = self.client.post(
            "/v1/images/generations",
            headers={
                "X-GreyFlow-API-Key": "test-greyflow-key",
            },
            json={"prompt": "A small red house"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("test-openai-key", response.text)


if __name__ == "__main__":
    unittest.main()
