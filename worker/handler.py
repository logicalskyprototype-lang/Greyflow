
import base64
import io
import secrets
import threading

import runpod
import torch
from diffusers import StableDiffusionPipeline
from pydantic import BaseModel, ConfigDict, Field, ValidationError


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MAX_PROMPT_LENGTH = 2000
MAX_STEPS = 35
MAX_DIMENSION = 512
MAX_OUTPUT_BYTES = 5 * 1024 * 1024

if not torch.cuda.is_available():
    raise RuntimeError("GreyFlow requires a CUDA-capable NVIDIA GPU.")

pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline = pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=True)

generation_lock = threading.Lock()


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)
    width: int = Field(default=512, ge=256, le=MAX_DIMENSION)
    height: int = Field(default=512, ge=256, le=MAX_DIMENSION)
    steps: int = Field(default=25, ge=1, le=MAX_STEPS)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


def handler(job):
    job_input = job.get("input")

    if not isinstance(job_input, dict):
        raise ValueError("Job input must be a JSON object.")

    try:
        request = ImageInput.model_validate(job_input)
    except ValidationError as exc:
        raise ValueError("Invalid image-generation input.") from exc

    if request.width % 64 != 0 or request.height % 64 != 0:
        raise ValueError("Width and height must be multiples of 64.")

    if not generation_lock.acquire(blocking=False):
        raise RuntimeError("Worker is busy; retry the job.")

    try:
        seed = (
            request.seed
            if request.seed is not None
            else secrets.randbelow(2**32)
        )

        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            result = pipeline(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                generator=generator,
            )

        buffer = io.BytesIO()
        result.images[0].save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        if len(image_bytes) > MAX_OUTPUT_BYTES:
            raise RuntimeError(
                "Generated image exceeds the worker output limit."
            )

        return {
            "job_id": str(job.get("id", "")),
            "status": "completed",
            "content_type": "image/png",
            "seed": seed,
            "width": request.width,
            "height": request.height,
            "image_base64": base64.b64encode(
                image_bytes
            ).decode("ascii"),
        }

    finally:
        generation_lock.release()


runpod.serverless.start({"handler": handler})
