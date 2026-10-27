import hmac
import json
import os
import re
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.image_validation import decode_valid_png
from app.runpod_contract import validate_worker_output


APP_NAME = "GreyFlow"
APP_VERSION = "0.2.0"

OPENAI_API_URL = "https://api.openai.com/v1/images/generations"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GREYFLOW_API_KEY = os.getenv("GREYFLOW_API_KEY", "").strip()
IMAGE_MODEL = os.getenv("GREYFLOW_IMAGE_MODEL", "gpt-image-1").strip()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
RUNPOD_IMAGE_ENDPOINT_ID = os.getenv(
    "RUNPOD_IMAGE_ENDPOINT_ID", ""
).strip()
RUNPOD_API_BASE = "https://api.runpod.ai/v2"

MAX_PROMPT_LENGTH = 4000
MAX_RUNPOD_PROMPT_LENGTH = 2000
MAX_RUNPOD_DIMENSION = 512
MAX_RUNPOD_STEPS = 35
MAX_RUNPOD_ERROR_LENGTH = 1000

# A 25 MiB image needs at most about 33.34 MiB of Base64.
# Leave limited room for the surrounding JSON response.
MAX_PROVIDER_RESPONSE_BYTES = 36 * 1024 * 1024

# RunPod worker images are limited to 5 MiB after Base64 decoding.
# Eight MiB accommodates the corresponding Base64 and JSON overhead.
MAX_RUNPOD_RESPONSE_BYTES = 8 * 1024 * 1024

RUNPOD_JOB_STATUSES = frozenset({
    "IN_QUEUE",
    "IN_PROGRESS",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMED_OUT",
})

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "GREYFLOW_ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title=APP_NAME,
    description="GreyFlow API for controlled AI task execution.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-GreyFlow-API-Key"],
)


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)
    size: Literal[
        "1024x1024",
        "1536x1024",
        "1024x1536",
    ] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Prompt must not be blank.")
        return value


class GeneratedImage(BaseModel):
    model: str
    size: str
    format: Literal["png"]
    data_url: str


class ImageGenerationResponse(BaseModel):
    status: Literal["completed"]
    images: list[GeneratedImage]


class RunPodImageJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(
        min_length=1,
        max_length=MAX_RUNPOD_PROMPT_LENGTH,
    )
    width: int = Field(default=512, ge=256, le=MAX_RUNPOD_DIMENSION)
    height: int = Field(default=512, ge=256, le=MAX_RUNPOD_DIMENSION)
    steps: int = Field(default=25, ge=1, le=MAX_RUNPOD_STEPS)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Prompt must not be blank.")
        return value

    @field_validator("width", "height")
    @classmethod
    def dimensions_must_be_multiples_of_64(
        cls, value: int
    ) -> int:
        if value % 64 != 0:
            raise ValueError("Dimensions must be multiples of 64.")
        return value


class RunPodJobSubmission(BaseModel):
    job_id: str
    status: str


class RunPodJobStatus(BaseModel):
    job_id: str
    status: str
    output: Any | None = None
    error: str | None = None


def require_api_key(supplied_key: str | None) -> None:
    if not GREYFLOW_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GreyFlow API authentication is not configured.",
        )

    if not isinstance(supplied_key, str) or not supplied_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing GreyFlow API key.",
        )

    # Compare bytes so non-ASCII input cannot cause compare_digest
    # to raise TypeError. Constant-time comparison is preserved.
    supplied_bytes = supplied_key.encode("utf-8")
    expected_bytes = GREYFLOW_API_KEY.encode("utf-8")

    if not hmac.compare_digest(supplied_bytes, expected_bytes):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing GreyFlow API key.",
        )


def require_runpod_config() -> tuple[str, str]:
    if not RUNPOD_API_KEY or not RUNPOD_IMAGE_ENDPOINT_ID:
        raise HTTPException(
            status_code=503,
            detail="RunPod image generation is not configured.",
        )

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}",
        RUNPOD_IMAGE_ENDPOINT_ID,
    ):
        raise HTTPException(
            status_code=503,
            detail="RunPod endpoint configuration is invalid.",
        )

    return RUNPOD_API_KEY, RUNPOD_IMAGE_ENDPOINT_ID


async def call_runpod(
    method: Literal["GET", "POST"],
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_key, endpoint_id = require_runpod_config()
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/{path}"
    timeout = httpx.Timeout(30.0, connect=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                method,
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload if method == "POST" else None,
            ) as response:
                response.raise_for_status()

                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError:
                        declared_length = None

                    if (
                        declared_length is not None
                        and declared_length > MAX_RUNPOD_RESPONSE_BYTES
                    ):
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "RunPod response exceeded the "
                                "allowed size."
                            ),
                        )

                body = bytearray()

                async for chunk in response.aiter_bytes():
                    if (
                        len(body) + len(chunk)
                        > MAX_RUNPOD_RESPONSE_BYTES
                    ):
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "RunPod response exceeded the "
                                "allowed size."
                            ),
                        )

                    body.extend(chunk)

                result = json.loads(body)

    except httpx.TimeoutException:
        if method == "POST":
            raise HTTPException(
                status_code=504,
                detail=(
                    "RunPod submission timed out. The job may have been "
                    "accepted. Check the RunPod endpoint's job history "
                    "before submitting again; do not retry automatically."
                ),
            ) from None

        raise HTTPException(
            status_code=504,
            detail=(
                "RunPod status check timed out. Retry the status check "
                "using the same job ID."
            ),
        ) from None

    except httpx.HTTPStatusError as exc:
        upstream_status = exc.response.status_code

        if upstream_status == 429:
            raise HTTPException(
                status_code=429,
                detail="RunPod rate limit reached. Try again later.",
            ) from None

        if upstream_status == 404 and method == "GET":
            raise HTTPException(
                status_code=404,
                detail="RunPod job was not found.",
            ) from None

        if upstream_status in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="RunPod rejected the server credentials or access.",
            ) from None

        raise HTTPException(
            status_code=502,
            detail="RunPod could not complete the API request.",
        ) from None

    except (httpx.RequestError, ValueError):
        if method == "POST":
            raise HTTPException(
                status_code=502,
                detail=(
                    "RunPod submission could not be confirmed. Check "
                    "RunPod job history before attempting another submission."
                ),
            ) from None

        raise HTTPException(
            status_code=502,
            detail="Could not retrieve the RunPod job status.",
        ) from None

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail="RunPod returned an invalid response.",
        )

    return result


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "message": "GreyFlow API is running.",
        "health": "/health",
        "image_endpoint": "/v1/images/generations",
        "runpod_submit_endpoint": "/v1/jobs/images",
        "runpod_status_endpoint": "/v1/jobs/images/{job_id}",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "image_generation_configured": bool(
            OPENAI_API_KEY and GREYFLOW_API_KEY
        ),
        "runpod_image_generation_configured": bool(
            RUNPOD_API_KEY and RUNPOD_IMAGE_ENDPOINT_ID
        ),
    }


@app.post(
    "/v1/images/generations",
    response_model=ImageGenerationResponse,
)
async def generate_image(
    request: ImageGenerationRequest,
    x_greyflow_api_key: str | None = Header(
        default=None,
        alias="X-GreyFlow-API-Key",
    ),
) -> ImageGenerationResponse:
    require_api_key(x_greyflow_api_key)

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Image provider credentials are not configured.",
        )

    provider_payload = {
        "model": IMAGE_MODEL,
        "prompt": request.prompt,
        "size": request.size,
        "quality": request.quality,
        "n": 1,
        "output_format": "png",
    }

    timeout = httpx.Timeout(120.0, connect=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=provider_payload,
            ) as response:
                response.raise_for_status()

                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError:
                        declared_length = None

                    if (
                        declared_length is not None
                        and declared_length > MAX_PROVIDER_RESPONSE_BYTES
                    ):
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "Image provider response exceeded "
                                "the allowed size."
                            ),
                        )

                body = bytearray()

                async for chunk in response.aiter_bytes():
                    if (
                        len(body) + len(chunk)
                        > MAX_PROVIDER_RESPONSE_BYTES
                    ):
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "Image provider response exceeded "
                                "the allowed size."
                            ),
                        )

                    body.extend(chunk)

                provider_result = json.loads(body)

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=(
                "Image generation timed out. Please check provider status "
                "before retrying."
            ),
        ) from None

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Image provider rate limit reached. Try again later.",
            ) from None

        raise HTTPException(
            status_code=502,
            detail="Image provider could not complete the request.",
        ) from None

    except (httpx.RequestError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Could not communicate with the image provider.",
        ) from None

    if not isinstance(provider_result, dict):
        raise HTTPException(
            status_code=502,
            detail="Image provider returned an invalid response.",
        )

    image_items = provider_result.get("data")

    if not isinstance(image_items, list) or len(image_items) != 1:
        raise HTTPException(
            status_code=502,
            detail="Image provider returned an unexpected number of images.",
        )

    item = image_items[0]

    if not isinstance(item, dict):
        raise HTTPException(
            status_code=502,
            detail="Image provider returned an invalid image item.",
        )

    encoded_image = item.get("b64_json")

    if not isinstance(encoded_image, str) or not encoded_image:
        raise HTTPException(
            status_code=502,
            detail="Image provider returned no usable image data.",
        )

    expected_width, expected_height = (
        int(part) for part in request.size.split("x")
    )
    expected_size = (expected_width, expected_height)

    image_bytes = decode_valid_png(encoded_image, expected_size)

    if image_bytes is None:
        raise HTTPException(
            status_code=502,
            detail="Provider response did not contain a valid PNG image.",
        )

    generated_image = GeneratedImage(
        model=IMAGE_MODEL,
        size=request.size,
        format="png",
        data_url=f"data:image/png;base64,{encoded_image}",
    )

    return ImageGenerationResponse(
        status="completed",
        images=[generated_image],
    )


@app.post(
    "/v1/jobs/images",
    response_model=RunPodJobSubmission,
    status_code=202,
)
async def submit_runpod_image_job(
    request: RunPodImageJobRequest,
    x_greyflow_api_key: str | None = Header(
        default=None,
        alias="X-GreyFlow-API-Key",
    ),
) -> RunPodJobSubmission:
    require_api_key(x_greyflow_api_key)
    require_runpod_config()

    payload = {
        "input": {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "seed": request.seed,
        }
    }

    result = await call_runpod("POST", "run", payload)

    job_id = result.get("id")
    status = result.get("status")

    if (
        not isinstance(job_id, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id)
        or not isinstance(status, str)
        or status not in RUNPOD_JOB_STATUSES
    ):
        raise HTTPException(
            status_code=502,
            detail="RunPod returned an invalid job submission response.",
        )

    return RunPodJobSubmission(job_id=job_id, status=status)


@app.get(
    "/v1/jobs/images/{job_id}",
    response_model=RunPodJobStatus,
)
async def get_runpod_image_job(
    job_id: str,
    x_greyflow_api_key: str | None = Header(
        default=None,
        alias="X-GreyFlow-API-Key",
    ),
) -> RunPodJobStatus:
    require_api_key(x_greyflow_api_key)

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid job ID.",
        )

    result = await call_runpod("GET", f"status/{job_id}")

    returned_id = result.get("id")
    status = result.get("status")

    if (
        returned_id != job_id
        or not isinstance(status, str)
        or status not in RUNPOD_JOB_STATUSES
    ):
        raise HTTPException(
            status_code=502,
            detail="RunPod returned an invalid job status response.",
        )

    error_value = result.get("error")
    if isinstance(error_value, str):
        error = error_value[:MAX_RUNPOD_ERROR_LENGTH]
    else:
        error = None

    # Never forward provider output for a job that has not completed.
    # Only completed output is eligible for validation and return.
    output = None

    if status == "COMPLETED":
        try:
            validated_output = validate_worker_output(
                output=result.get("output"),
                expected_job_id=job_id,
            )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=502,
                detail=(
                    "RunPod completed the job but returned "
                    "an invalid image result."
                ),
            ) from None

        output = validated_output.model_dump()

    return RunPodJobStatus(
        job_id=job_id,
        status=status,
        output=output,
        error=error,
    )
