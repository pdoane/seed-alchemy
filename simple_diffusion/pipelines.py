import gc
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline, DiffusionPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPipeline)
from image_metadata import ImageMetadata
from PIL import Image


@dataclass
class GenerateRequest:
    source_image: Image.Image = None
    controlnet_conditioning_image: Image.Image = None
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
        self.pipe = None
        self.dtype = torch.float32
        self.device = 'mps'

    @abstractmethod
    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
        pass

class ImagePipeline(PipelineBase):
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
            self.control_net = prev_pipeline.control_net
            self.control_net_model = image_metadata.control_net_model
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
                    print('Loading ControlNet Pipeline', image_metadata.model, image_metadata.control_net_model)
                    self.control_net = ControlNetModel.from_pretrained(image_metadata.control_net_model, torch_dtype=self.dtype)
                    self.control_net_model = image_metadata.control_net_model

                    if self.type == 'controlnet':
                        if image_metadata.safety_checker:
                            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=self.control_net, torch_dtype=self.dtype)
                        else:
                            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=self.control_net, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
                    elif self.type == 'controlnet_img2img':
                        if image_metadata.safety_checker:
                            self.pipe = DiffusionPipeline.from_pretrained(model, custom_pipeline='stable_diffusion_controlnet_img2img', controlnet=self.control_net, torch_dtype=self.dtype)
                        else:
                            self.pipe = DiffusionPipeline.from_pretrained(model, custom_pipeline='stable_diffusion_controlnet_img2img', controlnet=self.control_net, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)

            self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()

    def is_compatible(self, prev_pipeline: PipelineBase, image_metadata: ImageMetadata) -> bool:
        if not isinstance(prev_pipeline, ImagePipeline):
            return False
        if prev_pipeline.type != self.type:
            if self.type == 'txt2img' and prev_pipeline.type != 'img2img':
                return False
            elif self.type == 'img2img' and prev_pipeline.type != 'txt2img':
                return False
            else:
                return False
        if prev_pipeline.type == 'control_net' or prev_pipeline.type == 'controlnet_img2img':
            if prev_pipeline.control_net_model != image_metadata.control_net_model:
                return False
        
        return True

    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
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
                strength=req.image_metadata.img_strength,
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
                image=req.controlnet_conditioning_image,
                controlnet_conditioning_scale=req.image_metadata.control_net_scale,
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
                strength=req.image_metadata.img_strength,
                controlnet_conditioning_image=req.controlnet_conditioning_image,
                controlnet_conditioning_scale=req.image_metadata.control_net_scale,
                num_inference_steps=req.image_metadata.num_inference_steps,
                guidance_scale=req.image_metadata.guidance_scale,
                num_images_per_prompt=req.num_images_per_prompt,
                generator=req.generator,
                prompt_embeds=req.prompt_embeds,
                negative_prompt_embeds=req.negative_prompt_embeds,
                callback=req.callback,
            ).images
