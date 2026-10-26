import base64
import hmac
import os
import re
from typing import Any, Literal
import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    prompt: str = Field(min_length=1, max_length=4000)
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
    prompt: str = Field(min_length=1, max_length=2000)
    width: int = Field(default=512, ge=256, le=512)
    height: int = Field(default=512, ge=256, le=512)
    steps: int = Field(default=25, ge=1, le=35)
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
    def dimensions_must_be_multiples_of_64(cls, value: int) -> int:
        if value % 64 != 0:
            raise ValueError("Width and height must be multiples of 64.")
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
    if not supplied_key or not hmac.compare_digest(
        supplied_key, GREYFLOW_API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing GreyFlow API key.",
        )
def require_runpod_config() -> None:
    if not RUNPOD_API_KEY or not RUNPOD_IMAGE_ENDPOINT_ID:
        raise HTTPException(
            status_code=503,
            detail="RunPod image generation is not configured.",
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", RUNPOD_IMAGE_ENDPOINT_ID):
        raise HTTPException(
            status_code=503,
            detail="RunPod endpoint configuration is invalid.",
        )
async def call_runpod(
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = (
        "https://api.runpod.ai/v2/"
        f"{RUNPOD_IMAGE_ENDPOINT_ID}/{path}"
    )
    timeout = httpx.Timeout(30.0, connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="RunPod request timed out. Check the job status before retrying.",
        ) from None
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="RunPod rate limit reached. Try again later.",
            ) from None
        if exc.response.status_code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="RunPod rejected the server credentials.",
            ) from None
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="RunPod job or endpoint was not found.",
            ) from None
        raise HTTPException(
            status_code=502,
            detail="RunPod could not complete the request.",
        ) from None
    except (httpx.RequestError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Could not communicate with RunPod.",
        ) from None
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail="RunPod returned an invalid response.",
        )
    return result
@app.get("/")
async def root() -> dict[str, str]:
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
async def health() -> dict[str, object]:
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
            response = await client.post(
                OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=provider_payload,
            )
            response.raise_for_status()
            provider_result = response.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Image generation timed out. Please try again.",
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
    image_items = provider_result.get("data", [])
    if not isinstance(image_items, list) or not image_items:
        raise HTTPException(
            status_code=502,
            detail="Image provider returned no image.",
        )
    generated_images: list[GeneratedImage] = []
    for item in image_items:
        if not isinstance(item, dict):
            continue
        encoded_image = item.get("b64_json")
        if not isinstance(encoded_image, str) or not encoded_image:
            continue
        try:
            base64.b64decode(encoded_image, validate=True)
        except (ValueError, TypeError):
            continue
        generated_images.append(
            GeneratedImage(
                model=IMAGE_MODEL,
                size=request.size,
                format="png",
                data_url=f"data:image/png;base64,{encoded_image}",
            )
        )
    if not generated_images:
        raise HTTPException(
            status_code=502,
            detail="Provider response did not contain usable image data.",
        )
    return ImageGenerationResponse(
        status="completed",
        images=generated_images,
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
    result = await call_runpod("run", payload)
    job_id = result.get("id")
    status = result.get("status")
    if not isinstance(job_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}", job_id
    ):
        raise HTTPException(
            status_code=502,
            detail="RunPod returned an invalid job identifier.",
        )
    if not isinstance(status, str):
        raise HTTPException(
            status_code=502,
            detail="RunPod returned an invalid job status.",
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
    require_runpod_config()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid job identifier.",
        )
    result = await call_runpod(f"status/{job_id}")
    returned_id = result.get("id")
    status = result.get("status")
    if returned_id != job_id or not isinstance(status, str):
        raise HTTPException(
            status_code=502,
            detail="RunPod returned an invalid job status response.",
        )
    output = result.get("output")
    error = result.get("error")
    return RunPodJobStatus(
        job_id=job_id,
        status=status,
        output=output,
        error=error if isinstance(error, str) else None,
    )
