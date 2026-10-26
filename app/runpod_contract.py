import base64
import binascii
import io
import warnings
from typing import Any
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BASE64_LENGTH = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
MAX_IMAGE_DIMENSION = 512
MAX_IMAGE_PIXELS = MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
def _validate_png_bytes(image_bytes: bytes) -> tuple[int, int]:
    """Fully validate PNG contents and return the decoded dimensions."""
    if not image_bytes:
        raise ValueError("Image data must not be empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the maximum allowed size.")
    if not image_bytes.startswith(PNG_SIGNATURE):
        raise ValueError("Image data does not have a PNG signature.")
    try:
        # Treat Pillow's decompression-bomb warning as an error.
        # Check dimensions before loading pixel data.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.format != "PNG":
                    raise ValueError("Image data is not a PNG image.")
                width, height = image.size
                if width < 1 or height < 1:
                    raise ValueError("PNG dimensions must be positive.")
                if (
                    width > MAX_IMAGE_DIMENSION
                    or height > MAX_IMAGE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise ValueError(
                        "PNG dimensions exceed the maximum allowed size."
                    )
                # Verify file structure and checksums.
                image.verify()
            # Reopen after verify(), then decode the actual image data.
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.format != "PNG":
                    raise ValueError("Image data is not a PNG image.")
                if image.size != (width, height):
                    raise ValueError(
                        "PNG dimensions changed during validation."
                    )
                image.load()
        return width, height
    except ValueError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
    ):
        raise ValueError(
            "Image data is not a complete, valid PNG image."
        ) from None
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
        allowed = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
        )
        if not value or any(character not in allowed for character in value):
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
        # Bound the encoded input before allocating decoded image bytes.
        if len(value) > MAX_IMAGE_BASE64_LENGTH:
            raise ValueError("Image exceeds the maximum allowed size.")
        try:
            image_bytes = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError, TypeError):
            raise ValueError("Image data is not valid Base64.") from None
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the maximum allowed size.")
        _validate_png_bytes(image_bytes)
        return value
def validate_worker_output(
    output: Any,
    expected_job_id: str,
) -> RunPodImageOutput:
    """Validate a completed worker result and bind it to the requested job."""
    result = RunPodImageOutput.model_validate(output)
    if result.job_id != expected_job_id:
        raise ValueError(
            "Worker output job ID does not match the requested job."
        )
    image_bytes = base64.b64decode(
        result.image_base64,
        validate=True,
    )
    image_width, image_height = _validate_png_bytes(image_bytes)
    if image_width != result.width or image_height != result.height:
        raise ValueError(
            "PNG dimensions do not match the declared dimensions."
        )
    return result
