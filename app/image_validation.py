import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError

Keep provider-supplied image data within predictable resource limits.

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
PNG_SIGNATURE = b”\x89PNG\r\n\x1a\n”

Maximum Base64 characters needed to represent MAX_IMAGE_BYTES.

MAX_BASE64_CHARS = 4 * ((MAX_IMAGE_BYTES + 2) // 3)

def decode_valid_png(
encoded_image: str,
expected_size: tuple[int, int],
) -> bytes | None:
“”“Decode and validate a Base64 PNG without accepting oversized payloads.”””
if (
not isinstance(encoded_image, str)
or not encoded_image
or len(encoded_image) > MAX_BASE64_CHARS
):
return None

try:
    image_bytes = base64.b64decode(
        encoded_image,
        validate=True,
    )
except (binascii.Error, ValueError, TypeError):
    return None
if (
    not image_bytes
    or len(image_bytes) > MAX_IMAGE_BYTES
    or not image_bytes.startswith(PNG_SIGNATURE)
):
    return None
try:
    # Check file structure and dimensions before decoding pixel data.
    with Image.open(BytesIO(image_bytes)) as image:
        if image.format != "PNG":
            return None
        width, height = image.size
        if (
            width < 1
            or height < 1
            or (width, height) != expected_size
            or width * height > MAX_IMAGE_PIXELS
        ):
            return None
        image.verify()
    # verify() checks structure; load() confirms pixel data can be decoded.
    with Image.open(BytesIO(image_bytes)) as image:
        if image.format != "PNG":
            return None
        width, height = image.size
        if (
            width < 1
            or height < 1
            or (width, height) != expected_size
            or width * height > MAX_IMAGE_PIXELS
        ):
            return None
        image.load()
    return image_bytes
except (
    UnidentifiedImageError,
    Image.DecompressionBombError,
    OSError,
    ValueError,
    SyntaxError,
):
    return None
