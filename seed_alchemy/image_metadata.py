import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json
from PIL import Image

from . import configuration, utils, scheduler_registry


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
class InpaintMetadata:
    source: str = ""
    use_alpha_channel: bool = False
    invert_mask: bool = False


@dataclass_json
@dataclass
class ImageMetadata:
    mode: str = "image"
    model: str = "stabilityai/stable-diffusion-2-1-base"
    safety_checker: bool = True
    scheduler: str = "euler_a"
    path: str = ""  # TODO - remove as this is just for runtime convenience
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    width: int = 512
    height: int = 512
    img2img: Optional[Img2ImgMetadata] = None
    control_net: Optional[ControlNetMetadata] = None
    upscale: Optional[UpscaleMetadata] = None
    face: Optional[FaceRestorationMetadata] = None
    high_res: Optional[HighResMetadata] = None
    inpaint: Optional[InpaintMetadata] = None

    def load_from_params(self, data: dict):
        self.mode = data["mode"]
        self.model = data["model"]
        self.safety_checker = data["safety_checker"]
        self.scheduler = data["scheduler"]
        self.prompt = data["prompt"]
        self.negative_prompt = data["negative_prompt"]
        self.seed = data["seed"]
        self.num_inference_steps = data["num_inference_steps"]
        self.guidance_scale = data["guidance_scale"]
        self.width = data["width"]
        self.height = data["height"]

        if self.mode == "image":
            if data["img2img_enabled"]:
                self.img2img = Img2ImgMetadata(
                    source=data["img2img_source"],
                    noise=data["img2img_noise"],
                )
        elif self.mode == "canvas":
            self.img2img = Img2ImgMetadata(
                source=os.path.join(configuration.CANVAS_DIR, configuration.INPAINT_IMAGE_NAME),
                noise=data["inpaint_noise"],
            )

        if data["control_net_enabled"]:
            self.control_net = ControlNetMetadata(
                guidance_start=data["control_net_guidance_start"],
                guidance_end=data["control_net_guidance_end"],
                conditions=[ControlNetConditionMetadata.from_dict(item) for item in data["control_net_conditions"]],
            )

        if data["upscale_enabled"]:
            self.upscale = UpscaleMetadata(
                factor=data["upscale_factor"],
                denoising=data["upscale_denoising"],
                blend=data["upscale_blend"],
            )

        if data["face_enabled"]:
            self.face = FaceRestorationMetadata(blend=data["face_blend"])

        if data["high_res_enabled"]:
            self.high_res = HighResMetadata(
                factor=data["high_res_factor"],
                steps=data["high_res_steps"],
                guidance_scale=data["high_res_guidance_scale"],
                noise=data["high_res_noise"],
            )

        if self.mode == "image":
            if data["inpaint_enabled"]:
                self.inpaint = InpaintMetadata(
                    source=data["inpaint_source"],
                    use_alpha_channel=data["inpaint_use_alpha_channel"],
                    invert_mask=data["inpaint_invert_mask"],
                )
        elif self.mode == "canvas":
            self.inpaint = InpaintMetadata(
                source=os.path.join(configuration.CANVAS_DIR, configuration.INPAINT_IMAGE_NAME),
                use_alpha_channel=True,
                invert_mask=True,
            )

    def load_from_image(self, image: Image.Image):
        self.width = image.width
        self.height = image.height

        if "seed-alchemy" in image.info:
            metadata = vars(ImageMetadata.from_json(image.info["seed-alchemy"]))
            metadata.pop("path")
            self.__dict__.update(metadata)

        if self.scheduler not in scheduler_registry.DICT:
            self.scheduler = "euler_a"

    def save_to_png_info(self, png_info):
        filtered_dict = utils.remove_none_fields(self.to_dict())
        filtered_dict.pop("path")
        png_info.add_text("seed-alchemy", json.dumps(filtered_dict))
