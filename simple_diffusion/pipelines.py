import gc
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import configuration
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline, DiffusionPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPipeline)
from image_metadata import ImageMetadata
from PIL import Image


@dataclass
class GenerateRequest:
    source_image: Image.Image = None
    controlnet_conditioning_images: list[Image.Image] = field(default_factory=list)
    image_metadata: ImageMetadata = ImageMetadata()
    num_images_per_prompt: int = 1
    generator: torch.Generator = None
    prompt_embeds: torch.FloatTensor = None
    negative_prompt_embeds: torch.FloatTensor = None
    callback: Callable[[int, int, torch.FloatTensor], None] = None

class PipelineCache:
    def __init__(self):
        self.pipeline = None

class PipelineBase(ABC):
    def __init__(self) -> None:
        self.dtype = torch.float32
        self.device = 'mps'

    @abstractmethod
    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
        pass

class ImagePipeline(PipelineBase):
    type: str = None
    pipe: DiffusionPipeline = None
    control_nets: list[ControlNetModel] = []
    control_net_model_names: list[str] = []

    def __init__(self, pipeline_cache: PipelineCache, image_metadata: ImageMetadata) -> None:
        super().__init__()

        if image_metadata.control_net_enabled:
            if image_metadata.img2img_enabled:
                self.type = 'controlnet_img2img'
            else:
                self.type = 'controlnet'
        elif image_metadata.img2img_enabled:
            self.type = 'img2img'
        else:
            self.type = 'txt2img'

        prev_pipeline = pipeline_cache.pipeline
        if self.is_compatible(prev_pipeline, image_metadata):
            self.control_nets = prev_pipeline.control_nets
            self.control_net_model_names = prev_pipeline.control_net_model_names
            if self.type == prev_pipeline.type:
                self.pipe = prev_pipeline.pipe
            elif self.type == 'txt2img':
                self.pipe = StableDiffusionPipeline(**prev_pipeline.pipe.components, requires_safety_checker=False)
            elif self.type == 'img2img':
                self.pipe = StableDiffusionImg2ImgPipeline(**prev_pipeline.pipe.components, requires_safety_checker=False)
        else:
            prev_pipeline = None
            pipeline_cache.pipeline = None
            gc.collect()

            model = os.path.expanduser(image_metadata.model)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if self.type == 'txt2img':
                    print('Loading Stable Diffusion Pipeline', image_metadata.model)

                    if image_metadata.safety_checker:
                        self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=self.dtype)
                    else:
                        self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
                elif self.type == 'img2img':
                    print('Loading Stable Diffusion Pipeline', image_metadata.model)

                    if image_metadata.safety_checker:
                        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=self.dtype)
                    else:
                        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
                else:
                    control_net_model_names = []
                    for control_net_meta in image_metadata.control_nets:
                        control_net_model_names.append(control_net_meta.name)

                    print('Loading ControlNet Pipeline', image_metadata.model, control_net_model_names)

                    control_nets = []
                    for control_net_meta in image_metadata.control_nets:
                        control_net_config = configuration.control_net_models[control_net_meta.name]
                        control_net = ControlNetModel.from_pretrained(control_net_config.repo_id, subfolder=control_net_config.subfolder, torch_dtype=self.dtype)
                        control_nets.append(control_net)
                    
                    self.control_nets = control_nets
                    self.control_net_model_names = control_net_model_names

                    if len(self.control_nets) == 1:
                        self.control_nets = self.control_nets[0]

                    if self.type == 'controlnet':
                        if image_metadata.safety_checker:
                            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=self.control_nets, torch_dtype=self.dtype)
                        else:
                            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=self.control_nets, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
                    elif self.type == 'controlnet_img2img':
                        if image_metadata.safety_checker:
                            self.pipe = DiffusionPipeline.from_pretrained(model, custom_pipeline='stable_diffusion_controlnet_img2img', controlnet=self.control_nets, torch_dtype=self.dtype)
                        else:
                            self.pipe = DiffusionPipeline.from_pretrained(model, custom_pipeline='stable_diffusion_controlnet_img2img', controlnet=self.control_nets, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)

            self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()

    def is_compatible(self, prev_pipeline: PipelineBase, image_metadata: ImageMetadata) -> bool:
        if not isinstance(prev_pipeline, ImagePipeline):
            return False
        if self.type != prev_pipeline.type:
            if self.type == 'txt2img':
                if prev_pipeline.type != 'img2img':
                    return False
            elif self.type == 'img2img':
                if prev_pipeline.type != 'txt2img':
                    return False
            else:
                return False
        if self.type == 'controlnet' or self.type == 'controlnet_img2img':
            control_net_model_names = []
            for control_net_meta in image_metadata.control_nets:
                control_net_model_names.append(control_net_meta.name)
            if prev_pipeline.control_net_model_names != control_net_model_names:
                return False

        return True

    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
        control_net_scales = []
        for control_net_meta in req.image_metadata.control_nets:
            control_net_scales.append(control_net_meta.scale)

        if len(req.image_metadata.control_nets) == 1:
            controlnet_conditioning_images = req.controlnet_conditioning_images[0]
            control_net_scales = control_net_scales[0]
        else:
            controlnet_conditioning_images = req.controlnet_conditioning_images

        if self.type == 'txt2img':
            return self.pipe(
                width=req.image_metadata.width,
                height=req.image_metadata.height,
                num_inference_steps=req.image_metadata.num_inference_steps,
                guidance_scale=req.image_metadata.guidance_scale,
                num_images_per_prompt=req.num_images_per_prompt,
                generator=req.generator,
                prompt_embeds=req.prompt_embeds,
                negative_prompt_embeds=req.negative_prompt_embeds,
                callback=req.callback,
            ).images
        elif self.type == 'img2img':
            return self.pipe(
                image=req.source_image,
                strength=req.image_metadata.img2img_strength,
                num_inference_steps=req.image_metadata.num_inference_steps,
                guidance_scale=req.image_metadata.guidance_scale,
                num_images_per_prompt=req.num_images_per_prompt,
                generator=req.generator,
                prompt_embeds=req.prompt_embeds,
                negative_prompt_embeds=req.negative_prompt_embeds,
                callback=req.callback,
            ).images
        elif self.type == 'controlnet':
            return self.pipe(
                image=controlnet_conditioning_images,
                controlnet_conditioning_scale=control_net_scales,
                num_inference_steps=req.image_metadata.num_inference_steps,
                guidance_scale=req.image_metadata.guidance_scale,
                num_images_per_prompt=req.num_images_per_prompt,
                generator=req.generator,
                prompt_embeds=req.prompt_embeds,
                negative_prompt_embeds=req.negative_prompt_embeds,
                callback=req.callback,
            ).images
        elif self.type == 'controlnet_img2img':
            return self.pipe(
                image=req.source_image,
                strength=req.image_metadata.img2img_strength,
                controlnet_conditioning_image=controlnet_conditioning_images,
                controlnet_conditioning_scale=control_net_scales,
                controlnet_guidance_start=req.image_metadata.control_net_guidance_start,
                controlnet_guidance_end=req.image_metadata.control_net_guidance_end,
                num_inference_steps=req.image_metadata.num_inference_steps,
                guidance_scale=req.image_metadata.guidance_scale,
                num_images_per_prompt=req.num_images_per_prompt,
                generator=req.generator,
                prompt_embeds=req.prompt_embeds,
                negative_prompt_embeds=req.negative_prompt_embeds,
                callback=req.callback,
            ).images
