import gc
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPipeline)
from image_metadata import ImageMetadata
from PIL import Image


@dataclass
class GenerateRequest:
    source_image: Image.Image = None
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

class SDPipelineBase(PipelineBase):
    def __init__(self) -> None:
        super().__init__()

    def sd_load(self, pipeline_type: type, image_metadata: ImageMetadata):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = os.path.expanduser(image_metadata.model)
            if image_metadata.safety_checker:
                pipe = pipeline_type.from_pretrained(model, torch_dtype=self.dtype)
            else:
                pipe = pipeline_type.from_pretrained(model, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
            pipe.to(self.device)
            pipe.enable_attention_slicing()
        return pipe

class Txt2ImgPipeline(SDPipelineBase):
    def __init__(self, pipeline_cache: PipelineCache, image_metadata: ImageMetadata) -> None:
        super().__init__()
        prev_pipeline = pipeline_cache.pipeline
        if isinstance(prev_pipeline, Txt2ImgPipeline):
            self.pipe = prev_pipeline.pipe
        if isinstance(prev_pipeline, SDPipelineBase):
            self.pipe = StableDiffusionPipeline(**prev_pipeline.pipe.components, requires_safety_checker=False)
        else:
            prev_pipeline = None
            pipeline_cache.pipeline = None
            gc.collect()
            print('Loading Stable Diffusion Pipeline', image_metadata.model)
            self.pipe = self.sd_load(StableDiffusionPipeline, image_metadata)

    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
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


class Img2ImgPipeline(SDPipelineBase):
    def __init__(self, pipeline_cache: PipelineCache, image_metadata: ImageMetadata) -> None:
        super().__init__()
        prev_pipeline = pipeline_cache.pipeline
        if isinstance(prev_pipeline, Img2ImgPipeline):
            self.pipe = prev_pipeline.pipe
        if isinstance(prev_pipeline, SDPipelineBase):
            self.pipe = StableDiffusionImg2ImgPipeline(**prev_pipeline.pipe.components, requires_safety_checker=False)
        else:
            prev_pipeline = None
            pipeline_cache.pipeline = None
            gc.collect()
            print('Loading Stable Diffusion Pipeline', image_metadata.model)
            self.pipe = self.sd_load(StableDiffusionImg2ImgPipeline, image_metadata)

    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
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

class ControlNetPipeline(PipelineBase):
    def __init__(self, pipeline_cache: PipelineCache, image_metadata: ImageMetadata) -> None:
        super().__init__()
        prev_pipeline = pipeline_cache.pipeline
        if isinstance(prev_pipeline, ControlNetPipeline) and prev_pipeline.control_net_model == image_metadata.control_net_model:
            self.control_net = prev_pipeline.control_net
            self.control_net_model = image_metadata.control_net_model
            self.pipe = prev_pipeline.pipe
        else:
            prev_pipeline = None
            pipeline_cache.pipeline = None
            gc.collect()
            print('Loading ControlNet', image_metadata.model, image_metadata.control_net_model)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.control_net = ControlNetModel.from_pretrained(image_metadata.control_net_model, torch_dtype=self.dtype)
                self.control_net_model = image_metadata.control_net_model

                model = os.path.expanduser(image_metadata.model)
                if image_metadata.safety_checker:
                    self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=self.control_net, torch_dtype=self.dtype)
                else:
                    self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=self.control_net, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)
                self.pipe.to(self.device)
                self.pipe.enable_attention_slicing()

    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
        return self.pipe(
            image=req.source_image,
            num_inference_steps=req.image_metadata.num_inference_steps,
            guidance_scale=req.image_metadata.guidance_scale,
            num_images_per_prompt=req.num_images_per_prompt,
            generator=req.generator,
            prompt_embeds=req.prompt_embeds,
            negative_prompt_embeds=req.negative_prompt_embeds,
            callback=req.callback,
            controlnet_conditioning_scale=req.image_metadata.control_net_scale
        ).images
