import base64
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import main

class FakeResponse:
def init(self, payload, status_code=200):
self.payload = payload
self.status_code = status_code
self.request = httpx.Request(
“POST”,
“https://provider.test/request”,
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
“GREYFLOW_API_KEY”: main.GREYFLOW_API_KEY,
“OPENAI_API_KEY”: main.OPENAI_API_KEY,
“IMAGE_MODEL”: main.IMAGE_MODEL,
“RUNPOD_API_KEY”: main.RUNPOD_API_KEY,
“RUNPOD_IMAGE_ENDPOINT_ID”: (
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
    self.assertTrue(
        call["url"].endswith("/test-endpoint/run")
    )
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
def test_runpod_status_uses_get_and_returns_output(self):
    FakeAsyncClient.next_payload = {
        "id": "job_123",
        "status": "COMPLETED",
        "output": {
            "job_id": "job_123",
            "status": "completed",
            "content_type": "image/png",
            "seed": 123,
            "width": 512,
            "height": 512,
            "image_base64": "ZmFrZS1pbWFnZQ==",
        },
    }
    response = self.client.get(
        "/v1/jobs/images/job_123",
        headers=self.headers(),
    )
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()["job_id"], "job_123")
    self.assertEqual(response.json()["status"], "COMPLETED")
    self.assertEqual(
        response.json()["output"]["content_type"],
        "image/png",
    )
    call = FakeAsyncClient.calls[0]
    self.assertEqual(call["method"], "GET")
    self.assertTrue(
        call["url"].endswith("/test-endpoint/status/job_123")
    )
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
def test_runpod_status_rejects_mismatched_job_id(self):
    FakeAsyncClient.next_payload = {
        "id": "different_job",
        "status": "COMPLETED",
        "output": {},
    }
    response = self.client.get(
        "/v1/jobs/images/job_123",
        headers=self.headers(),
    )
    self.assertEqual(response.status_code, 502)
def test_openai_image_endpoint_preserves_response_format(self):
    encoded = base64.b64encode(b"fake-png-content").decode("ascii")
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
    self.assertEqual(call["json"]["output_format"], "png")

if name == “main”:
unittest.main()
