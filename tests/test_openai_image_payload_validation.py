
import base64
import io
import struct
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app import main


def valid_png_bytes(width=1024, height=1024):
    """Create a complete PNG image in memory."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


class FakeResponse:
    status_code = 200
    encoded_image = base64.b64encode(
        b"this is not an image"
    ).decode("ascii")

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"b64_json": self.encoded_image}]}


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
        FakeResponse.encoded_image = base64.b64encode(
            b"this is not an image"
        ).decode("ascii")

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

    def post_image_request(self, size="1024x1024"):
        return self.client.post(
            "/v1/images/generations",
            headers={
                "X-GreyFlow-API-Key": "test-greyflow-key",
            },
            json={
                "prompt": "A small red house",
                "size": size,
            },
        )

    def assert_payload_rejected(self):
        response = self.post_image_request()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("test-openai-key", response.text)

    def test_valid_base64_that_is_not_an_image_is_rejected(self):
        FakeResponse.encoded_image = base64.b64encode(
            b"this is not an image"
        ).decode("ascii")

        self.assert_payload_rejected()

    def test_truncated_png_is_rejected(self):
        # PNG signature and an incomplete IHDR chunk are not a complete file.
        truncated_png = (
            b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">II", 1024, 1024)
        )
        FakeResponse.encoded_image = base64.b64encode(
            truncated_png
        ).decode("ascii")

        self.assert_payload_rejected()

    def test_valid_png_with_wrong_dimensions_is_rejected(self):
        # The request defaults to 1024x1024, but this PNG is 512x512.
        FakeResponse.encoded_image = base64.b64encode(
            valid_png_bytes(width=512, height=512)
        ).decode("ascii")

        self.assert_payload_rejected()

    def test_valid_png_with_matching_dimensions_is_accepted(self):
        FakeResponse.encoded_image = base64.b64encode(
            valid_png_bytes(width=1024, height=1024)
        ).decode("ascii")

        response = self.post_image_request()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(len(response.json()["images"]), 1)


if __name__ == "__main__":
    unittest.main()
