import gc
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import configuration
import torch
from compel.embeddings_provider import BaseTextualInversionManager
from diffusers import ControlNetModel, DiffusionPipeline
from diffusers.loaders import TextualInversionLoaderMixin
from image_metadata import ImageMetadata
from PIL import Image
from stable_diffusion_pipeline import StableDiffusionPipeline


class DiffusersTextualInversionManager(BaseTextualInversionManager):
    def __init__(self, pipe):
        self.pipe = pipe

    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) == 0:
            return token_ids

        prompt = self.pipe.tokenizer.decode(token_ids)
        prompt = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
        return self.pipe.tokenizer.encode(prompt, add_special_tokens=False)

@dataclass
class GenerateRequest:
    source_image: Image.Image = None
    controlnet_conditioning_images: list[Image.Image] = field(default_factory=list)
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
    def __init__(self) -> None:
        self.dtype = torch.float32
        self.device = 'mps'

    @abstractmethod
    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
        pass

class ImagePipeline(PipelineBase):
    type: str = None
    pipe: DiffusionPipeline = None
    model: str = None
    control_nets: list[ControlNetModel] = []
    control_net_model_names: list[str] = []
    textual_inversion_manager: DiffusersTextualInversionManager = None

    def __init__(self, pipeline_cache: PipelineCache, image_metadata: ImageMetadata) -> None:
        super().__init__()

        prev_pipeline = pipeline_cache.pipeline

        # Control nets
        prev_control_nets = {}
        if prev_pipeline:
            for name, control_net in zip(prev_pipeline.control_net_names, prev_pipeline.control_nets):
                prev_control_nets[name] = control_net

        self.control_nets = []
        self.control_net_names = []
        for control_net_meta in image_metadata.control_nets:
            control_net_config = configuration.control_net_models[control_net_meta.name]
            if control_net_config.subfolder is not None:
                control_net_name = '{:s}/{:s}'.format(control_net_config.repo_id, control_net_config.subfolder)
            else:
                control_net_name = control_net_config.repo_id

            if control_net_name in prev_control_nets:
                control_net = prev_control_nets[control_net_name]
            else:
                print('Loading ControlNet', control_net_name)
                control_net = ControlNetModel.from_pretrained(control_net_config.repo_id, subfolder=control_net_config.subfolder, torch_dtype=self.dtype)
                control_net.to(self.device)
                control_net.set_attention_slice('auto')

            self.control_nets.append(control_net)
            self.control_net_names.append(control_net_name)

        # Pipeline
        self.model = os.path.expanduser(image_metadata.model)

        if isinstance(prev_pipeline, ImagePipeline) and prev_pipeline.model == self.model:
            self.pipe = prev_pipeline.pipe
            self.textual_inversion_manager = prev_pipeline.textual_inversion_manager
        else:
            prev_pipeline = None
            pipeline_cache.pipeline = None
            gc.collect()

            # Model
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                print('Loading Stable Diffusion Pipeline', image_metadata.model)

                if image_metadata.safety_checker:
                    self.pipe = StableDiffusionPipeline.from_pretrained(self.model, torch_dtype=self.dtype)
                else:
                    self.pipe = StableDiffusionPipeline.from_pretrained(self.model, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False)

            self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()

            # Textual Inversion
            if isinstance(self.pipe, TextualInversionLoaderMixin):
                self.textual_inversion_manager = DiffusersTextualInversionManager(self.pipe)
                for entry in configuration.known_embeddings:
                    entry_path = configuration.get_embedding_path(entry)
                    name, _ = os.path.splitext(entry)
                    print('Loading textual inversion', name)
                    self.pipe.load_textual_inversion(entry_path, name)

    def __call__(self, req: GenerateRequest) -> list[Image.Image]:
        # Image parameters
        image = None
        img2img_strength = 1.0
        if req.image_metadata.img2img_enabled:
            image = req.source_image
            img2img_strength = req.image_metadata.img2img_strength

        # Controlnet parameters
        controlnet = None
        controlnet_conditioning_image = None
        controlnet_conditioning_scale = 1.0
        controlnet_guidance_start = 0.0
        controlnet_guidance_end = 1.0
        if req.image_metadata.control_net_enabled:
            controlnet_conditioning_scale = []
            for control_net_meta in req.image_metadata.control_nets:
                controlnet_conditioning_scale.append(control_net_meta.scale)
            controlnet_guidance_start = req.image_metadata.control_net_guidance_start
            controlnet_guidance_end = req.image_metadata.control_net_guidance_end

            if len(req.image_metadata.control_nets) == 1:
                controlnet = self.control_nets[0]
                controlnet_conditioning_image = req.controlnet_conditioning_images[0]
                controlnet_conditioning_scale = controlnet_conditioning_scale[0]
            else:
                controlnet = self.control_nets
                controlnet_conditioning_image = req.controlnet_conditioning_images

        return self.pipe(
            width=req.image_metadata.width,
            height=req.image_metadata.height,
            image=image,
            strength=img2img_strength,
            controlnet=controlnet,
            controlnet_conditioning_image=controlnet_conditioning_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            controlnet_guidance_start=controlnet_guidance_start,
            controlnet_guidance_end=controlnet_guidance_end,
            num_inference_steps=req.image_metadata.num_inference_steps,
            guidance_scale=req.image_metadata.guidance_scale,
            num_images_per_prompt=req.num_images_per_prompt,
            generator=req.generator,
            prompt_embeds=req.prompt_embeds,
            negative_prompt_embeds=req.negative_prompt_embeds,
            callback=req.callback,
        ).images
