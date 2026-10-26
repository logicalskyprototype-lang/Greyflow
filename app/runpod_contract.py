
import base64
import binascii
import struct
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_IMAGE_BYTES = 5 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RunPodImageOutput(BaseModel):
    """Validated output contract for a completed GreyFlow image job."""

    model_config = ConfigDict(extra="forbid", strict=True)

    job_id: str = Field(min_length=1, max_length=128)
    status: str
    content_type: str
    seed: int = Field(ge=0, le=2**32 - 1)
    width: int = Field(ge=256, le=512)
    height: int = Field(ge=256, le=512)
    image_base64: str = Field(min_length=1)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if not value or any(
            character not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in value
        ):
            raise ValueError("Invalid worker job ID.")
        return value

    @field_validator("status")
    @classmethod
    def validate_completed_status(cls, value: str) -> str:
        if value != "completed":
            raise ValueError("Image output must have completed status.")
        return value

    @field_validator("content_type")
    @classmethod
    def validate_png_content_type(cls, value: str) -> str:
        if value != "image/png":
            raise ValueError("Only PNG image output is supported.")
        return value

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        if value % 64 != 0:
            raise ValueError("Image dimensions must be multiples of 64.")
        return value

    @field_validator("image_base64")
    @classmethod
    def validate_png_data(cls, value: str) -> str:
        try:
            image_bytes = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("Image data is not valid Base64.") from None

        if not image_bytes:
            raise ValueError("Image data must not be empty.")

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the maximum allowed size.")

        if not image_bytes.startswith(PNG_SIGNATURE):
            raise ValueError("Image data does not have a PNG signature.")

        if len(image_bytes) < 24:
            raise ValueError("PNG image data is incomplete.")

        # PNG IHDR stores the image dimensions at fixed byte offsets.
        png_width, png_height = struct.unpack(">II", image_bytes[16:24])

        if png_width != 0 and png_width != 0:
            # Dimensions are checked against the worker's declared values
            # by validate_worker_output() below.
            pass

        return value


def validate_worker_output(
    output: Any,
    expected_job_id: str,
) -> RunPodImageOutput:
    """Validate a completed worker result and bind it to the requested job."""

    result = RunPodImageOutput.model_validate(output)

    if result.job_id != expected_job_id:
        raise ValueError("Worker output job ID does not match the requested job.")

    image_bytes = base64.b64decode(result.image_base64, validate=True)
    image_width, image_height = struct.unpack(">II", image_bytes[16:24])

    if image_width != result.width or image_height != result.height:
        raise ValueError("PNG dimensions do not match the declared dimensions.")

    if image_width < 1 or image_height < 1:
        raise ValueError("PNG dimensions must be positive.")

    return result
