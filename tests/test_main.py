
import base64
import struct
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import main


def png_header_base64(width=512, height=512):
    """Build a minimal PNG header fixture for contract validation."""
    png_header = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height)
    )
    return base64.b64encode(png_header).decode("ascii")


def valid_worker_output(
    job_id="job_123",
    width=512,
    height=512,
):
    return {
        "job_id": job_id,
        "status": "completed",
        "content_type": "image/png",
        "seed": 123,
        "width": width,
        "height": height,
        "image_base64": png_header_base64(width, height),
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.request = httpx.Request(
            "POST",
            "https://provider.test/request",
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
        return self.payload


class FakeAsyncClient:
    next_payload = {}
    next_status_code = 200
    calls = []

    def __init__(self, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append(
            {
                "method": "POST",
                "url": url,
                "headers": headers or {},
                "json": json,
            }
        )
        return FakeResponse(
            self.next_payload,
            self.next_status_code,
        )

    async def get(self, url, headers=None):
        self.calls.append(
            {
                "method": "GET",
                "url": url,
                "headers": headers or {},
            }
        )
        return FakeResponse(
            self.next_payload,
            self.next_status_code,
        )


class GreyFlowApiTests(unittest.TestCase):
    def setUp(self):
        self.original_values = {
            "GREYFLOW_API_KEY": main.GREYFLOW_API_KEY,
            "OPENAI_API_KEY": main.OPENAI_API_KEY,
            "IMAGE_MODEL": main.IMAGE_MODEL,
            "RUNPOD_API_KEY": main.RUNPOD_API_KEY,
            "RUNPOD_IMAGE_ENDPOINT_ID": (
                main.RUNPOD_IMAGE_ENDPOINT_ID
            ),
        }

        main.GREYFLOW_API_KEY = "test-greyflow-key"
        main.OPENAI_API_KEY = "test-openai-key"
        main.IMAGE_MODEL = "test-image-model"
        main.RUNPOD_API_KEY = "test-runpod-key"
        main.RUNPOD_IMAGE_ENDPOINT_ID = "test-endpoint"

        FakeAsyncClient.next_payload = {}
        FakeAsyncClient.next_status_code = 200
        FakeAsyncClient.calls = []

        self.provider_patch = patch.object(
            main.httpx,
            "AsyncClient",
            FakeAsyncClient,
        )
        self.provider_patch.start()
        self.addCleanup(self.provider_patch.stop)

        self.client = TestClient(main.app)

    def tearDown(self):
        for name, value in self.original_values.items():
            setattr(main, name, value)

    def headers(self):
        return {
            "X-GreyFlow-API-Key": "test-greyflow-key",
        }

    def get_job_status(self):
        return self.client.get(
            "/v1/jobs/images/job_123",
            headers=self.headers(),
        )

    def test_health_reports_service_and_version(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "GreyFlow")
        self.assertEqual(response.json()["version"], "0.2.0")
        self.assertTrue(
            response.json()["runpod_image_generation_configured"]
        )

    def test_image_endpoint_requires_api_key(self):
        response = self.client.post(
            "/v1/images/generations",
            json={"prompt": "A small red house"},
        )

        self.assertEqual(response.status_code, 401)

    def test_runpod_submission_sends_expected_payload(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "IN_QUEUE",
        }

        response = self.client.post(
            "/v1/jobs/images",
            headers=self.headers(),
            json={
                "prompt": "A small red house",
                "width": 512,
                "height": 512,
                "steps": 25,
                "seed": 123,
            },
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json(),
            {
                "job_id": "job_123",
                "status": "IN_QUEUE",
            },
        )

        call = FakeAsyncClient.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertTrue(call["url"].endswith("/test-endpoint/run"))
        self.assertEqual(
            call["headers"]["Authorization"],
            "Bearer test-runpod-key",
        )
        self.assertEqual(
            call["json"]["input"],
            {
                "prompt": "A small red house",
                "width": 512,
                "height": 512,
                "steps": 25,
                "seed": 123,
            },
        )

    def test_completed_runpod_job_returns_validated_output(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "COMPLETED",
            "output": valid_worker_output(),
        }

        response = self.get_job_status()

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["job_id"], "job_123")
        self.assertEqual(body["status"], "COMPLETED")
        self.assertEqual(body["output"]["job_id"], "job_123")
        self.assertEqual(body["output"]["content_type"], "image/png")
        self.assertEqual(body["output"]["width"], 512)
        self.assertEqual(body["output"]["height"], 512)

        call = FakeAsyncClient.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertTrue(
            call["url"].endswith(
                "/test-endpoint/status/job_123"
            )
        )

    def test_completed_job_rejects_invalid_base64(self):
        output = valid_worker_output()
        output["image_base64"] = "not-valid-base64!"

        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "COMPLETED",
            "output": output,
        }

        response = self.get_job_status()

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "RunPod completed the job but returned an invalid image result.",
        )

    def test_completed_job_rejects_non_png_data(self):
        output = valid_worker_output()
        output["image_base64"] = base64.b64encode(
            b"this is not a PNG file"
        ).decode("ascii")

        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "COMPLETED",
            "output": output,
        }

        response = self.get_job_status()

        self.assertEqual(response.status_code, 502)

    def test_completed_job_rejects_mismatched_png_dimensions(self):
        output = valid_worker_output(width=512, height=512)
        output["image_base64"] = png_header_base64(
            width=256,
            height=512,
        )

        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "COMPLETED",
            "output": output,
        }

        response = self.get_job_status()

        self.assertEqual(response.status_code, 502)

    def test_completed_job_rejects_worker_job_id_mismatch(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "COMPLETED",
            "output": valid_worker_output(job_id="different_job"),
        }

        response = self.get_job_status()

        self.assertEqual(response.status_code, 502)

    def test_queued_job_does_not_require_image_output(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "IN_QUEUE",
            "output": None,
        }

        response = self.get_job_status()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "IN_QUEUE")
        self.assertIsNone(response.json()["output"])

    def test_runpod_rejects_invalid_dimensions(self):
        response = self.client.post(
            "/v1/jobs/images",
            headers=self.headers(),
            json={
                "prompt": "A small red house",
                "width": 300,
                "height": 512,
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(FakeAsyncClient.calls, [])

    def test_runpod_rejects_extra_input_fields(self):
        response = self.client.post(
            "/v1/jobs/images",
            headers=self.headers(),
            json={
                "prompt": "A small red house",
                "unexpected_field": "not allowed",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(FakeAsyncClient.calls, [])

    def test_runpod_missing_configuration_returns_503(self):
        main.RUNPOD_API_KEY = ""

        response = self.client.post(
            "/v1/jobs/images",
            headers=self.headers(),
            json={"prompt": "A small red house"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(FakeAsyncClient.calls, [])

    def test_runpod_status_rejects_mismatched_outer_job_id(self):
        FakeAsyncClient.next_payload = {
            "id": "different_job",
            "status": "COMPLETED",
            "output": {},
        }

        response = self.get_job_status()

        self.assertEqual(response.status_code, 502)

    def test_openai_image_endpoint_preserves_response_format(self):
        # Use the shared PNG-header fixture rather than fake text bytes.
        encoded = png_header_base64()

        FakeAsyncClient.next_payload = {
            "data": [{"b64_json": encoded}],
        }

        response = self.client.post(
            "/v1/images/generations",
            headers=self.headers(),
            json={
                "prompt": "A small red house",
                "size": "1024x1024",
                "quality": "medium",
            },
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(
            body["images"][0]["data_url"],
            f"data:image/png;base64,{encoded}",
        )

        call = FakeAsyncClient.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(
            call["json"]["model"],
            "test-image-model",
        )
        self.assertEqual(
            call["json"]["output_format"],
            "png",
        )


if __name__ == "__main__":
    unittest.main()
