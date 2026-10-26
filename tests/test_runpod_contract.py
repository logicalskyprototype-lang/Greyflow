
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.runpod_contract import (
    MAX_IMAGE_BASE64_LENGTH,
    RunPodImageOutput,
)


class RunPodImageContractTests(unittest.TestCase):
    def test_oversized_base64_is_rejected_before_decoding(self):
        oversized_image = "A" * (MAX_IMAGE_BASE64_LENGTH + 1)

        output = {
            "job_id": "job_123",
            "status": "completed",
            "content_type": "image/png",
            "seed": 123,
            "width": 512,
            "height": 512,
            "image_base64": oversized_image,
        }

        with patch(
            "app.runpod_contract.base64.b64decode"
        ) as decoder:
            with self.assertRaises(ValidationError):
                RunPodImageOutput.model_validate(output)

            decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
