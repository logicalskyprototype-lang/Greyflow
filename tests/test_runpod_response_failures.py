import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import main


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
    next_exception = None
    calls = []

    def __init__(self, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append(
            ("POST", url, headers or {}, json)
        )

        if self.next_exception is not None:
            raise self.next_exception

        return FakeResponse(
            self.next_payload,
            self.next_status_code,
        )

    async def get(self, url, headers=None):
        self.calls.append(
            ("GET", url, headers or {}, None)
        )

        if self.next_exception is not None:
            raise self.next_exception

        return FakeResponse(
            self.next_payload,
            self.next_status_code,
        )


class RunPodResponseFailureTests(unittest.TestCase):
    def setUp(self):
        self.original_values = {
            "GREYFLOW_API_KEY": main.GREYFLOW_API_KEY,
            "RUNPOD_API_KEY": main.RUNPOD_API_KEY,
            "RUNPOD_IMAGE_ENDPOINT_ID": (
                main.RUNPOD_IMAGE_ENDPOINT_ID
            ),
        }

        main.GREYFLOW_API_KEY = "test-greyflow-key"
        main.RUNPOD_API_KEY = "test-runpod-key"
        main.RUNPOD_IMAGE_ENDPOINT_ID = "test-endpoint"

        FakeAsyncClient.next_payload = {}
        FakeAsyncClient.next_status_code = 200
        FakeAsyncClient.next_exception = None
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

    def submit_job(self):
        return self.client.post(
            "/v1/jobs/images",
            headers=self.headers(),
            json={"prompt": "A small red house"},
        )

    def check_job_status(self):
        return self.client.get(
            "/v1/jobs/images/job_123",
            headers=self.headers(),
        )

    def test_submission_rejects_non_object_responses(self):
        for payload in (None, [], "unexpected response"):
            with self.subTest(payload=payload):
                FakeAsyncClient.next_payload = payload

                response = self.submit_job()

                self.assertEqual(response.status_code, 502)

    def test_submission_rejects_missing_job_id(self):
        FakeAsyncClient.next_payload = {
            "status": "IN_QUEUE",
        }

        response = self.submit_job()

        self.assertEqual(response.status_code, 502)

    def test_submission_rejects_missing_status(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
        }

        response = self.submit_job()

        self.assertEqual(response.status_code, 502)

    def test_submission_rejects_invalid_job_id(self):
        FakeAsyncClient.next_payload = {
            "id": "job/123",
            "status": "IN_QUEUE",
        }

        response = self.submit_job()

        self.assertEqual(response.status_code, 502)

    def test_submission_rejects_unknown_status(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "UNRECOGNIZED_STATUS",
        }

        response = self.submit_job()

        self.assertEqual(response.status_code, 502)

    def test_status_response_rejects_missing_status(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
        }

        response = self.check_job_status()

        self.assertEqual(response.status_code, 502)

    def test_status_response_rejects_unknown_status(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "UNRECOGNIZED_STATUS",
        }

        response = self.check_job_status()

        self.assertEqual(response.status_code, 502)

    def test_status_response_rejects_mismatched_job_id(self):
        FakeAsyncClient.next_payload = {
            "id": "different_job",
            "status": "IN_PROGRESS",
        }

        response = self.check_job_status()

        self.assertEqual(response.status_code, 502)

    def test_lifecycle_statuses_are_returned_correctly(self):
        statuses = (
            "IN_QUEUE",
            "IN_PROGRESS",
            "FAILED",
            "CANCELLED",
            "TIMED_OUT",
        )

        for status in statuses:
            with self.subTest(status=status):
                FakeAsyncClient.next_payload = {
                    "id": "job_123",
                    "status": status,
                    "error": (
                        "Mock provider failure"
                        if status == "FAILED"
                        else None
                    ),
                }

                response = self.check_job_status()

                self.assertEqual(
                    response.status_code,
                    200,
                    msg=response.text,
                )
                self.assertEqual(
                    response.json()["job_id"],
                    "job_123",
                )
                self.assertEqual(
                    response.json()["status"],
                    status,
                )

    def test_failed_job_preserves_provider_error(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "FAILED",
            "error": "Mock provider failure",
        }

        response = self.check_job_status()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["status"],
            "FAILED",
        )
        self.assertEqual(
            response.json()["error"],
            "Mock provider failure",
        )

    def test_submission_timeout_returns_504(self):
        FakeAsyncClient.next_exception = httpx.ReadTimeout(
            "Mock submission timeout"
        )

        response = self.submit_job()

        self.assertEqual(response.status_code, 504)
        self.assertIn(
            "may have been accepted",
            response.json()["detail"],
        )
        self.assertIn(
            "do not retry automatically",
            response.json()["detail"],
        )

    def test_status_polling_timeout_returns_504(self):
        FakeAsyncClient.next_exception = httpx.ReadTimeout(
            "Mock status timeout"
        )

        response = self.check_job_status()

        self.assertEqual(response.status_code, 504)
        self.assertIn(
            "same job ID",
            response.json()["detail"],
        )

    def test_submission_provider_server_error_returns_502(self):
        FakeAsyncClient.next_status_code = 500
        FakeAsyncClient.next_payload = {
            "error": "Mock provider error",
        }

        response = self.submit_job()

        self.assertEqual(response.status_code, 502)

    def test_missing_runpod_configuration_returns_503(self):
        main.RUNPOD_API_KEY = ""

        response = self.submit_job()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(FakeAsyncClient.calls, [])


if __name__ == "__main__":
    unittest.main()
