import json
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import main


class FakeResponse:
    def __init__(
        self,
        payload,
        status_code=200,
        json_exception=None,
        body=None,
        headers=None,
        stream_exception=None,
        stream_read_exception=None,
    ):
        self.payload = payload
        self.status_code = status_code
        self.json_exception = json_exception
        self.stream_exception = stream_exception
        self.stream_read_exception = stream_read_exception

        if body is not None:
            self.body = body
        elif json_exception is not None:
            self.body = b"{malformed provider JSON"
        else:
            self.body = json.dumps(payload).encode("utf-8")

        self.headers = (
            {"Content-Length": str(len(self.body))}
            if headers is None
            else headers
        )
        self.request = httpx.Request(
            "POST",
            "https://provider.test/request",
        )

    async def __aenter__(self):
        if self.stream_exception is not None:
            raise self.stream_exception
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

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

    async def aiter_bytes(self):
        chunk_size = 64 * 1024

        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

            # Simulate a transport failure only after a chunk
            # has already been delivered to the application.
            if self.stream_read_exception is not None:
                raise self.stream_read_exception


class FakeAsyncClient:
    next_payload = {}
    next_status_code = 200
    next_exception = None
    next_stream_read_exception = None
    next_json_exception = None
    next_body = None
    next_headers = None
    calls = []

    def __init__(self, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def stream(self, method, url, headers=None, json=None):
        method = method.upper()
        self.calls.append((method, url, headers or {}, json))

        return FakeResponse(
            payload=self.next_payload,
            status_code=self.next_status_code,
            json_exception=self.next_json_exception,
            body=self.next_body,
            headers=self.next_headers,
            stream_exception=self.next_exception,
            stream_read_exception=self.next_stream_read_exception,
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
        FakeAsyncClient.next_stream_read_exception = None
        FakeAsyncClient.next_json_exception = None
        FakeAsyncClient.next_body = None
        FakeAsyncClient.next_headers = None
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
        FakeAsyncClient.next_payload = {"status": "IN_QUEUE"}
        response = self.submit_job()
        self.assertEqual(response.status_code, 502)

    def test_submission_rejects_missing_status(self):
        FakeAsyncClient.next_payload = {"id": "job_123"}
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
        FakeAsyncClient.next_payload = {"id": "job_123"}
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
        self.assertEqual(response.json()["status"], "FAILED")
        self.assertEqual(
            response.json()["error"],
            "Mock provider failure",
        )

    def test_oversized_content_length_is_rejected(self):
        FakeAsyncClient.next_payload = {
            "id": "job_123",
            "status": "IN_QUEUE",
        }
        FakeAsyncClient.next_headers = {
            "Content-Length": str(
                main.MAX_RUNPOD_RESPONSE_BYTES + 1
            ),
        }

        for request_name in ("submit", "poll"):
            with self.subTest(request=request_name):
                response = (
                    self.submit_job()
                    if request_name == "submit"
                    else self.check_job_status()
                )
                self.assertEqual(response.status_code, 502)

    def test_streamed_response_over_limit_is_rejected(self):
        FakeAsyncClient.next_headers = {}
        FakeAsyncClient.next_body = (
            b"x" * (main.MAX_RUNPOD_RESPONSE_BYTES + 1)
        )

        for request_name in ("submit", "poll"):
            with self.subTest(request=request_name):
                response = (
                    self.submit_job()
                    if request_name == "submit"
                    else self.check_job_status()
                )
                self.assertEqual(response.status_code, 502)

    def test_submission_timeout_returns_504_without_auto_retry(self):
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
        self.assertEqual(
            sum(call[0] == "POST" for call in FakeAsyncClient.calls),
            1,
        )

    def test_status_polling_timeout_returns_504(self):
        FakeAsyncClient.next_exception = httpx.ReadTimeout(
            "Mock status timeout"
        )

        response = self.check_job_status()

        self.assertEqual(response.status_code, 504)
        self.assertIn("same job ID", response.json()["detail"])

    def test_submission_provider_server_error_returns_502(self):
        FakeAsyncClient.next_status_code = 500
        FakeAsyncClient.next_payload = {"error": "Mock provider error"}

        response = self.submit_job()

        self.assertEqual(response.status_code, 502)

    def test_rate_limit_returns_429_for_submission_and_polling(self):
        for request_name in ("submit", "poll"):
            with self.subTest(request=request_name):
                FakeAsyncClient.next_status_code = 429

                response = (
                    self.submit_job()
                    if request_name == "submit"
                    else self.check_job_status()
                )

                self.assertEqual(response.status_code, 429)
                self.assertIn(
                    "rate limit",
                    response.json()["detail"].lower(),
                )

    def test_missing_job_during_polling_returns_404(self):
        FakeAsyncClient.next_status_code = 404

        response = self.check_job_status()

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"],
            "RunPod job was not found.",
        )

    def test_provider_auth_failures_return_502_without_leaking_secrets(
        self,
    ):
        for status_code in (401, 403):
            with self.subTest(status_code=status_code):
                FakeAsyncClient.next_status_code = status_code

                response = self.submit_job()

                self.assertEqual(response.status_code, 502)
                self.assertNotIn("test-runpod-key", response.text)
                self.assertNotIn("Bearer", response.text)

    def test_malformed_provider_json_returns_502(self):
        for request_name in ("submit", "poll"):
            with self.subTest(request=request_name):
                FakeAsyncClient.next_json_exception = ValueError(
                    "Malformed provider JSON"
                )

                response = (
                    self.submit_job()
                    if request_name == "submit"
                    else self.check_job_status()
                )

                self.assertEqual(response.status_code, 502)
                self.assertNotIn(
                    "Malformed provider JSON",
                    response.text,
                )

    def test_connection_failure_returns_502(self):
        request = httpx.Request(
            "POST",
            "https://provider.test/request",
        )
        FakeAsyncClient.next_exception = httpx.ConnectError(
            "Mock connection failure",
            request=request,
        )

        response = self.submit_job()

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("Mock connection failure", response.text)
        self.assertNotIn("test-runpod-key", response.text)

    def test_interrupted_stream_returns_safe_502(self):
        for request_name in ("submit", "poll"):
            with self.subTest(request=request_name):
                FakeAsyncClient.next_payload = {
                    "id": "job_123",
                    "status": "IN_QUEUE",
                }

                # Force at least two chunks so the read error occurs
                # after the first chunk has been yielded.
                FakeAsyncClient.next_body = (
                    json.dumps(
                        FakeAsyncClient.next_payload
                    ).encode("utf-8")
                    + b" " * (64 * 1024)
                )
                FakeAsyncClient.next_headers = {}

                method = "POST" if request_name == "submit" else "GET"
                request = httpx.Request(
                    method,
                    "https://provider.test/request",
                )
                FakeAsyncClient.next_stream_read_exception = (
                    httpx.ReadError(
                        "Mock interrupted response stream",
                        request=request,
                    )
                )

                response = (
                    self.submit_job()
                    if request_name == "submit"
                    else self.check_job_status()
                )

                self.assertEqual(response.status_code, 502)
                self.assertNotIn(
                    "Mock interrupted response stream",
                    response.text,
                )
                self.assertNotIn("test-runpod-key", response.text)
                self.assertNotIn("Bearer", response.text)

                FakeAsyncClient.next_stream_read_exception = None
                FakeAsyncClient.next_body = None
                FakeAsyncClient.next_headers = None

    def test_missing_runpod_configuration_returns_503(self):
        main.RUNPOD_API_KEY = ""

        response = self.submit_job()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(FakeAsyncClient.calls, [])


if __name__ == "__main__":
    unittest.main()
