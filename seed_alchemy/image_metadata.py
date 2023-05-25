import json
from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json
from PIL import Image

from . import utils


@dataclass_json
@dataclass
class ControlNetConditionMetadata:
    model: str = ""
    source: str = ""
    preprocessor: Optional[str] = None
    params: Optional[List[float]] = field(default_factory=list)
    scale: float = 1.0


@dataclass_json
@dataclass
class Img2ImgMetadata:
    source: str = ""
    noise: float = 0.5


@dataclass_json
@dataclass
class ControlNetMetadata:
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    conditions: list[ControlNetConditionMetadata] = field(default_factory=list)


@dataclass_json
@dataclass
class UpscaleMetadata:
    factor: int = 1
    denoising: float = 0.75
    blend: float = 0.75


@dataclass_json
@dataclass
class FaceRestorationMetadata:
    blend: float = 0.75


@dataclass_json
@dataclass
class HighResMetadata:
    factor: float = 1.5
    steps: int = 20
    guidance_scale: float = 7.0
    noise: float = 0.5


@dataclass_json
@dataclass
class ImageMetadata:
    model: str = "stabilityai/stable-diffusion-2-1-base"
    safety_checker: bool = True
    scheduler: str = "k_euler_a"
    path: str = ""  # TODO - remove as this is just for runtime convenience
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 1
    num_inference_steps: int = 20
    guidance_scale: float = 7.0
    width: int = 512
    height: int = 512
    img2img: Optional[Img2ImgMetadata] = None
    control_net: Optional[ControlNetMetadata] = None
    upscale: Optional[UpscaleMetadata] = None
    face: Optional[FaceRestorationMetadata] = None
    high_res: Optional[HighResMetadata] = None

    def load_from_settings(self, settings):
        self.model = settings.value("model")
        self.safety_checker = settings.value("safety_checker", type=bool)
        self.scheduler = settings.value("scheduler", type=str)
        self.prompt = settings.value("prompt", type=str)
        self.negative_prompt = settings.value("negative_prompt", type=str)

        self.seed = settings.value("seed", type=int)
        self.num_inference_steps = settings.value("num_inference_steps", type=int)
        self.guidance_scale = settings.value("guidance_scale", type=float)
        self.width = settings.value("width", type=int)
        self.height = settings.value("height", type=int)

        if settings.value("img2img_enabled", type=bool):
            self.img2img = Img2ImgMetadata(
                source=settings.value("img2img_source", type=str),
                noise=settings.value("img2img_noise", type=float),
            )

        if settings.value("control_net_enabled", type=bool):
            self.control_net = ControlNetMetadata(
                guidance_start=settings.value("control_net_guidance_start", type=float),
                guidance_end=settings.value("control_net_guidance_end", type=float),
                conditions=[
                    ControlNetConditionMetadata.from_dict(item)
                    for item in json.loads(settings.value("control_net_conditions"))
                ],
            )

        if settings.value("upscale_enabled", type=bool):
            self.upscale = UpscaleMetadata(
                factor=settings.value("upscale_factor", type=int),
                denoising=settings.value("upscale_denoising", type=float),
                blend=settings.value("upscale_blend", type=float),
            )

        if settings.value("face_enabled", type=bool):
            self.face = FaceRestorationMetadata(blend=settings.value("face_blend", type=float))

        if settings.value("high_res_enabled", type=bool):
            self.high_res = HighResMetadata(
                factor=settings.value("high_res_factor", type=float),
                steps=settings.value("high_res_steps", type=int),
                guidance_scale=settings.value("high_res_guidance_scale", type=float),
                noise=settings.value("high_res_noise", type=float),
            )

    def load_from_image(self, image: Image.Image):
        self.width = image.width
        self.height = image.height

        if "seed-alchemy" in image.info:
            metadata = vars(ImageMetadata.from_json(image.info["seed-alchemy"]))
            metadata.pop("path")
            self.__dict__.update(metadata)

    def save_to_png_info(self, png_info):
        filtered_dict = utils.remove_none_fields(self.to_dict())
        filtered_dict.pop("path")
        png_info.add_text("seed-alchemy", json.dumps(filtered_dict))
