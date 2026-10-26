
import os
import re
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from diffusers import StableDiffusionPipeline
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


WORKER_API_KEY = os.getenv("GREYFLOW_WORKER_API_KEY", "").strip()
MODEL_ID = os.getenv(
    "GREYFLOW_SD_MODEL_ID",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
).strip()

OUTPUT_DIR = Path(
    os.getenv("GREYFLOW_OUTPUT_DIR", "/data/generated")
).resolve()

MAX_PROMPT_LENGTH = 2000
MAX_STEPS = 35
MAX_DIMENSION = 768

pipeline = None
generation_lock = threading.Lock()


class ImageRequest(BaseModel):
    job_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)
    width: int = Field(default=512, ge=256, le=MAX_DIMENSION)
    height: int = Field(default=512, ge=256, le=MAX_DIMENSION)
    steps: int = Field(default=25, ge=1, le=MAX_STEPS)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class ImageOutput(BaseModel):
    job_id: str
    status: str
    content_type: str
    output: dict[str, str]


def require_worker_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not WORKER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Worker authentication is not configured.",
        )

    expected = f"Bearer {WORKER_API_KEY}"
    if authorization is None or not secrets.compare_digest(
        authorization, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid worker credentials.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    if not WORKER_API_KEY:
        raise RuntimeError(
            "GREYFLOW_WORKER_API_KEY must be configured."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA-capable NVIDIA GPU is required for this worker."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    yield

    pipeline = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="GreyFlow Image Worker",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "service": "greyflow-image-worker",
        "status": "ready" if pipeline is not None else "starting",
        "gpu_required": True,
    }


@app.post(
    "/v1/worker/images",
    response_model=ImageOutput,
    dependencies=[Depends(require_worker_key)],
)
def generate_image(request: ImageRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Image model is not ready.",
        )

    if request.width % 64 != 0 or request.height % 64 != 0:
        raise HTTPException(
            status_code=422,
            detail="Width and height must be multiples of 64.",
        )

    # Restrict each request to a single inference at a time.
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Worker is busy. Retry later.",
        )

    try:
        seed = request.seed
        generator = torch.Generator(device="cuda")

        if seed is None:
            seed = secrets.randbelow(2**32)

        generator.manual_seed(seed)

        with torch.inference_mode():
            result = pipeline(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                generator=generator,
            )

        image = result.images[0]

        # Never use a customer-provided path or filename.
        filename = f"{request.job_id}.png"
        destination = (OUTPUT_DIR / filename).resolve()

        if destination.parent != OUTPUT_DIR:
            raise HTTPException(
                status_code=400,
                detail="Invalid output identifier.",
            )

        image.save(destination, format="PNG")

        return ImageOutput(
            job_id=request.job_id,
            status="completed",
            content_type="image/png",
            output={
                "storage_key": f"generated/{filename}",
            },
        )

    except HTTPException:
        raise
    except Exception:
        # Avoid exposing model paths, prompts, or internal errors.
        raise HTTPException(
            status_code=500,
            detail="Image generation failed.",
        ) from None
    finally:
        generation_lock.release()
