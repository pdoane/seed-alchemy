from enum import Enum
from uuid import UUID
from dataclasses import field
from typing import Optional
from pydantic import BaseModel

from enum import Enum
from typing import Optional


class PreviewType(str, Enum):
    LATENT = "latent"
    TINY_VAE = "tiny_vae"


class CancelRequest(BaseModel):
    session_id: UUID


class Img2ImgParams(BaseModel):
    source: str
    noise: float = 0.5


class LoraModelParams(BaseModel):
    model: str
    weight: float


class LoraParams(BaseModel):
    entries: list[LoraModelParams]


class ControlNetCondition(BaseModel):
    model: str
    source: str
    processor: str = "none"
    params: dict[str, float] = []
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    scale: float = 1.0


class ControlNetParams(BaseModel):
    conditions: list[ControlNetCondition]


class RefinerParams(BaseModel):
    model: str = "stable-diffusion-xl-refiner-1.0"
    cfg_scale: float = 4.0
    high_noise_end: Optional[float]
    steps: Optional[int]
    noise: Optional[float]


class UpscaleParams(BaseModel):
    factor: int
    denoising: float = 0.75
    blend: float = 0.75


class FaceRestorationParams(BaseModel):
    blend: float = 0.75


class HighResParams(BaseModel):
    factor: float = 1.5
    steps: int = 20
    cfg_scale: float = 4.0
    noise: float = 0.5


class InpaintParams(BaseModel):
    source: str = ""
    use_alpha_channel: bool = False
    invert_mask: bool = False


class ImageRequest(BaseModel):
    session_id: Optional[UUID] = None
    generator_id: Optional[UUID] = None
    user: str = "default"
    collection: str = "outputs"
    image_count: int = 1
    preview: Optional[PreviewType] = PreviewType.LATENT

    model: str = "stable-diffusion-v1-5"
    scheduler: str = "euler_a"
    safety_checker: bool = True
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    cfg_scale: float = 4.0
    width: int = 512
    height: int = 512
    seed: int = 1
    img2img: Optional[Img2ImgParams] = None
    lora: Optional[LoraParams] = None
    control_net: Optional[ControlNetParams] = None
    refiner: Optional[RefinerParams] = None
    upscale: Optional[UpscaleParams] = None
    face: Optional[FaceRestorationParams] = None
    high_res: Optional[HighResParams] = None
    inpaint: Optional[InpaintParams] = None


class PromptGenRequest(BaseModel):
    model: str = "promptgen-lexart"
    prompt: str = ""
    temperature: float = 1.0
    top_k: int = 12
    top_p: float = 1.0
    beam_count: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    min_length: int = 20
    max_length: int = 150
    count: int = 5
    seed: int = 1


class ProcessRequest(BaseModel):
    user: str
    collection: str = "outputs"
    source: str
    processor: str = "none"
    params: dict[str, float] = {}


class PathRequest(BaseModel):
    user: str
    path: str


class MoveRequest(BaseModel):
    user: str
    src_path: str
    dst_collection: str
