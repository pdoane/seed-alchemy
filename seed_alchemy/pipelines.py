import gc
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import torch
from diffusers import ControlNetModel
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.pipelines import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from PIL import Image

from . import configuration, control_net_config, lora
from .image_metadata import ImageMetadata


@dataclass
class GenerateRequest:
    collection: str = ""
    reduce_memory: bool = False
    source_image: Image.Image = None
    mask_image: Image.Image = None
    control_images: list[Image.Image] = field(default_factory=list)
    image_metadata: ImageMetadata = None
    num_images_per_prompt: int = 1
    generator: torch.Generator = None
    prompt_embeds: torch.FloatTensor = None
    negative_prompt_embeds: torch.FloatTensor = None
    callback: Callable[[int, int, torch.FloatTensor], None] = None


class PipelineCache:
    def __init__(self):
        self.pipeline = None


class PipelineBase(ABC):
    @abstractmethod
    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
        pass


class ImagePipeline(PipelineBase):
    def __init__(self, pipeline_cache: PipelineCache, image_metadata: ImageMetadata) -> None:
        super().__init__()
        self.pipe: StableDiffusionPipeline = None
        self.scheduler_config: dict = None
        self.model: str = None
        self.control_nets: list[ControlNetModel] = []
        self.control_net_model_names: list[str] = []

        prev_pipeline = pipeline_cache.pipeline

        # Control nets
        prev_control_nets = {}
        if prev_pipeline:
            for name, control_net in zip(prev_pipeline.control_net_names, prev_pipeline.control_nets):
                prev_control_nets[name] = control_net

        self.control_nets = []
        self.control_net_names = []

        control_net_meta = image_metadata.control_net
        if control_net_meta:
            for condition_meta in control_net_meta.conditions:
                control_net_model_config = control_net_config.models[condition_meta.model]
                if control_net_model_config.subfolder is not None:
                    control_net_name = "{:s}/{:s}".format(
                        control_net_model_config.repo_id, control_net_model_config.subfolder
                    )
                else:
                    control_net_name = control_net_model_config.repo_id

                if control_net_name in prev_control_nets:
                    control_net = prev_control_nets[control_net_name]
                else:
                    print("Loading ControlNet", control_net_name)
                    control_net = ControlNetModel.from_pretrained(
                        control_net_model_config.repo_id,
                        subfolder=control_net_model_config.subfolder,
                        torch_dtype=configuration.torch_dtype,
                    )
                    control_net.to(configuration.torch_device)
                    control_net.set_attention_slice("auto")

                self.control_nets.append(control_net)
                self.control_net_names.append(control_net_name)

        # Pipeline
        self.model = configuration.stable_diffusion_models[image_metadata.model]

        if isinstance(prev_pipeline, ImagePipeline) and prev_pipeline.model == self.model:
            self.pipe = prev_pipeline.pipe
            self.scheduler_config = prev_pipeline.scheduler_config
        else:
            prev_pipeline = None
            pipeline_cache.pipeline = None
            gc.collect()

            # Model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print("Loading Stable Diffusion Pipeline", image_metadata.model)

                if image_metadata.safety_checker:
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        self.model, torch_dtype=configuration.torch_dtype
                    )
                else:
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        self.model,
                        torch_dtype=configuration.torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                    )

            self.scheduler_config = self.pipe.scheduler.config.copy()

            self.pipe.to(configuration.torch_device)
            self.pipe.enable_attention_slicing()
            self.pipe.backup_weights = {}

            # Textual Inversion
            if isinstance(self.pipe, TextualInversionLoaderMixin):
                paths = []
                tokens = []
                for name, path in configuration.textual_inversions.items():
                    paths.append(path)
                    tokens.append(name)

                if paths is not []:
                    print("Loading Textual Inversions")
                    self.pipe.load_textual_inversion(paths, tokens)

    def set_loras(self, loras):
        lora_models = []
        lora_multipliers = []
        for lora_model, lora_weight in loras:
            path = configuration.loras.get(lora_model)
            if path:
                lora_models.append(lora.load(path, configuration.torch_device, configuration.torch_dtype))
                lora_multipliers.append(lora_weight)

        lora.apply(self.pipe, lora_models, lora_multipliers)

    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
        img2img_meta = req.image_metadata.img2img
        control_net_meta = req.image_metadata.control_net
        inpaint_meta = req.image_metadata.inpaint

        controlnet = None
        controlnet_conditioning_image = None
        controlnet_conditioning_scale = 1.0
        controlnet_guidance_start = 0.0
        controlnet_guidance_end = 1.0
        if control_net_meta:
            controlnet_conditioning_scale = []
            for condition_meta in control_net_meta.conditions:
                controlnet_conditioning_scale.append(condition_meta.scale)
            controlnet_guidance_start = control_net_meta.guidance_start
            controlnet_guidance_end = control_net_meta.guidance_end

            if len(control_net_meta.conditions) == 1:
                controlnet = self.control_nets[0]
                controlnet_conditioning_image = req.control_images[0]
                controlnet_conditioning_scale = controlnet_conditioning_scale[0]
            else:
                controlnet = self.control_nets
                controlnet_conditioning_image = req.control_images

        if inpaint_meta:
            return StableDiffusionInpaintPipeline(**self.pipe.components, requires_safety_checker=False)(
                callback=req.callback,
                generator=req.generator,
                guidance_scale=req.image_metadata.guidance_scale,
                height=req.image_metadata.height,
                image=req.source_image,
                mask_image=req.mask_image,
                negative_prompt_embeds=req.negative_prompt_embeds,
                num_images_per_prompt=req.num_images_per_prompt,
                num_inference_steps=req.image_metadata.num_inference_steps,
                prompt_embeds=req.prompt_embeds,
                strength=img2img_meta.noise,
                width=req.image_metadata.width,
            ).images
        elif img2img_meta:
            if control_net_meta:
                return StableDiffusionControlNetImg2ImgPipeline(
                    **self.pipe.components, controlnet=controlnet, requires_safety_checker=False
                )(
                    callback=req.callback,
                    control_image=controlnet_conditioning_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=req.generator,
                    guidance_scale=req.image_metadata.guidance_scale,
                    image=req.source_image,
                    negative_prompt_embeds=req.negative_prompt_embeds,
                    num_images_per_prompt=req.num_images_per_prompt,
                    num_inference_steps=req.image_metadata.num_inference_steps,
                    prompt_embeds=req.prompt_embeds,
                    strength=img2img_meta.noise,
                ).images
            else:
                return StableDiffusionImg2ImgPipeline(**self.pipe.components, requires_safety_checker=False)(
                    callback=req.callback,
                    generator=req.generator,
                    guidance_scale=req.image_metadata.guidance_scale,
                    image=req.source_image,
                    negative_prompt_embeds=req.negative_prompt_embeds,
                    num_images_per_prompt=req.num_images_per_prompt,
                    num_inference_steps=req.image_metadata.num_inference_steps,
                    prompt_embeds=req.prompt_embeds,
                    strength=img2img_meta.noise,
                ).images
        else:
            if control_net_meta:
                return StableDiffusionControlNetPipeline(
                    **self.pipe.components, controlnet=controlnet, requires_safety_checker=False
                )(
                    callback=req.callback,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=req.generator,
                    guidance_scale=req.image_metadata.guidance_scale,
                    height=req.image_metadata.height,
                    image=controlnet_conditioning_image,
                    negative_prompt_embeds=req.negative_prompt_embeds,
                    num_images_per_prompt=req.num_images_per_prompt,
                    num_inference_steps=req.image_metadata.num_inference_steps,
                    prompt_embeds=req.prompt_embeds,
                    width=req.image_metadata.width,
                ).images
            else:
                return StableDiffusionPipeline(**self.pipe.components, requires_safety_checker=False)(
                    callback=req.callback,
                    generator=req.generator,
                    guidance_scale=req.image_metadata.guidance_scale,
                    height=req.image_metadata.height,
                    negative_prompt_embeds=req.negative_prompt_embeds,
                    num_images_per_prompt=req.num_images_per_prompt,
                    num_inference_steps=req.image_metadata.num_inference_steps,
                    prompt_embeds=req.prompt_embeds,
                    width=req.image_metadata.width,
                ).images
