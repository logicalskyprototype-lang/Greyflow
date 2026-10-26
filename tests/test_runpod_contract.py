import base64
import io
import unittest
from unittest.mock import patch
from PIL import Image
from pydantic import ValidationError
from app.runpod_contract import (
    MAX_IMAGE_BASE64_LENGTH,
    RunPodImageOutput,
    validate_worker_output,
)
def make_png(width: int, height: int) -> bytes:
    """Build a complete, valid PNG image for contract tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()
def make_worker_output(
    image_bytes: bytes | None = None,
) -> dict:
    """Build a valid-shaped completed worker response."""
    if image_bytes is None:
        image_bytes = make_png(256, 256)
    return {
        "job_id": "job_123",
        "status": "completed",
        "content_type": "image/png",
        "seed": 123,
        "width": 256,
        "height": 256,
        "image_base64": base64.b64encode(
            image_bytes
        ).decode("ascii"),
    }
class RunPodImageContractTests(unittest.TestCase):
    def test_oversized_base64_is_rejected_before_decoding(self):
        output = make_worker_output()
        output["image_base64"] = "A" * (
            MAX_IMAGE_BASE64_LENGTH + 1
        )
        with patch(
            "app.runpod_contract.base64.b64decode"
        ) as decoder:
            with self.assertRaises(ValidationError):
                RunPodImageOutput.model_validate(output)
            decoder.assert_not_called()
    def test_non_png_signature_is_rejected(self):
        output = make_worker_output(b"not-a-png-file")
        with self.assertRaises(ValidationError):
            RunPodImageOutput.model_validate(output)
    def test_truncated_png_header_is_rejected(self):
        output = make_worker_output(
            b"\x89PNG\r\n\x1a\n"
        )
        with self.assertRaises(ValidationError):
            RunPodImageOutput.model_validate(output)
    def test_truncated_png_with_valid_header_is_rejected(self):
        valid_png = make_png(256, 256)
        # Keep the PNG signature and IHDR, but remove most image data.
        truncated_png = valid_png[:40]
        output = make_worker_output(truncated_png)
        with self.assertRaises(ValidationError):
            validate_worker_output(
                output,
                expected_job_id="job_123",
            )
    def test_png_dimensions_must_match_declared_dimensions(self):
        output = make_worker_output(make_png(256, 320))
        with self.assertRaisesRegex(
            ValueError,
            "PNG dimensions do not match",
        ):
            validate_worker_output(
                output,
                expected_job_id="job_123",
            )
    def test_unexpected_worker_response_shapes_are_rejected(self):
        for output in (None, [], "unexpected response"):
            with self.subTest(output=output):
                with self.assertRaises(ValidationError):
                    validate_worker_output(
                        output,
                        expected_job_id="job_123",
                    )
    def test_worker_job_id_must_match_requested_job(self):
        output = make_worker_output()
        with self.assertRaisesRegex(ValueError, "job ID"):
            validate_worker_output(
                output,
                expected_job_id="job_456",
            )
    def test_valid_png_is_accepted(self):
        output = make_worker_output(make_png(256, 256))
        result = validate_worker_output(
            output,
            expected_job_id="job_123",
        )
        self.assertEqual(result.job_id, "job_123")
        self.assertEqual(result.width, 256)
        self.assertEqual(result.height, 256)
if __name__ == "__main__":
    unittest.main()
