import gc
import os
from typing import Callable, Optional, Union

import torch
from compel import Compel, ReturnedEmbeddingsType
from compel.diffusers_textual_inversion_manager import DiffusersTextualInversionManager
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.loaders import TextualInversionLoaderMixin
from PIL import Image

from . import config, lora, scheduler_registry
from .device import default_device, default_dtype
from .models import ControlNetParams, LoraModelParams
from .types import BaseModelType


class UniversalPipeline:
    def __init__(self):
        self.device = default_device()
        self.torch_dtype = default_dtype()

        self.pipe = None
        self.model = None
        self.base_model_type = None
        self.safety_checker = None
        self.scheduler_config = None
        self.compel = None
        self.compel2 = None
        self.control_nets: list[ControlNetModel] = []
        self.control_net_names: list[str] = []

    def __call__(
        self,
        image_count: int,
        prompt: str,
        negative_prompt: str,
        steps: int,
        denoising_start: Optional[float],
        denoising_end: Optional[float],
        cfg_scale: float,
        width: int,
        height: int,
        generator: torch.Generator,
        noise: Optional[float],
        source_image: Optional[Union[Image.Image, torch.FloatTensor]],
        mask_image: Optional[Union[Image.Image, torch.FloatTensor]],
        control_net: Optional[ControlNetParams],
        control_images: Optional[list[Union[Image.Image, torch.FloatTensor]]],
        output_type: str,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]],
    ):
        # Step adjustment
        # if source_image is not None and noise is not None:
        #     diffusers_steps = int(steps * noise)
        #     scaled_steps = int(diffusers_steps / noise)
        #     while steps != int(scaled_steps * noise):
        #         scaled_steps += 1
        #     steps = scaled_steps

        # Prompt
        if self.base_model_type == BaseModelType.SDXL:
            # TODO - expose 2nd prompt
            prompt2 = prompt
            negative_prompt2 = negative_prompt

            prompt1_embeds = self.compel(prompt)
            prompt2_embeds, pooled_prompt_embeds = self.compel2(prompt2)
            prompt_embeds = torch.cat((prompt1_embeds, prompt2_embeds), dim=-1)

            negative_prompt1_embeds = self.compel(negative_prompt)
            negative_prompt2_embeds, negative_pooled_prompt_embeds = self.compel2(negative_prompt2)
            negative_prompt_embeds = torch.cat((negative_prompt1_embeds, negative_prompt2_embeds), dim=-1)

        elif self.base_model_type == BaseModelType.SDXL_REFINER:
            prompt_embeds, pooled_prompt_embeds = self.compel(prompt)
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.compel(negative_prompt)

        else:
            prompt_embeds = self.compel(prompt)
            negative_prompt_embeds = self.compel(negative_prompt)
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None

        # Strength
        strength = noise or 0.0

        if control_net is not None:
            controlnet_conditioning_scales = [condition.scale for condition in control_net.conditions]

            if len(control_net.conditions) == 1:
                controlnet = self.control_nets[0]
                control_image = control_images[0]
                controlnet_conditioning_scale = controlnet_conditioning_scales[0]
            else:
                controlnet = self.control_nets
                control_image = control_images
                controlnet_conditioning_scale = controlnet_conditioning_scales

            if mask_image is not None:
                return StableDiffusionControlNetInpaintPipeline(
                    **self.pipe.components,
                    controlnet=controlnet,
                    requires_safety_checker=False,
                )(
                    callback=callback,
                    control_image=control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator,
                    guidance_scale=cfg_scale,
                    height=height,
                    image=source_image,
                    mask_image=mask_image,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_images_per_prompt=image_count,
                    num_inference_steps=steps,
                    output_type=output_type,
                    prompt_embeds=prompt_embeds,
                    strength=strength,
                    width=width,
                ).images
            elif source_image is not None:
                return StableDiffusionControlNetImg2ImgPipeline(
                    **self.pipe.components,
                    controlnet=controlnet,
                    requires_safety_checker=False,
                )(
                    callback=callback,
                    control_image=control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator,
                    guidance_scale=cfg_scale,
                    image=source_image,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_images_per_prompt=image_count,
                    num_inference_steps=steps,
                    output_type=output_type,
                    prompt_embeds=prompt_embeds,
                    strength=strength,
                ).images
            else:
                if self.base_model_type == BaseModelType.SDXL:
                    return StableDiffusionXLControlNetPipeline(
                        **self.pipe.components,
                        controlnet=controlnet,
                    )(
                        callback=callback,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        # denoising_end=denoising_end,
                        generator=generator,
                        guidance_scale=cfg_scale,
                        height=height,
                        image=control_image,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_images_per_prompt=image_count,
                        num_inference_steps=steps,
                        output_type=output_type,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        prompt_embeds=prompt_embeds,
                        width=width,
                    ).images
                else:
                    return StableDiffusionControlNetPipeline(
                        **self.pipe.components,
                        controlnet=controlnet,
                        requires_safety_checker=False,
                    )(
                        callback=callback,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        generator=generator,
                        guidance_scale=cfg_scale,
                        height=height,
                        image=control_image,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_images_per_prompt=image_count,
                        num_inference_steps=steps,
                        output_type=output_type,
                        prompt_embeds=prompt_embeds,
                        width=width,
                    ).images

        else:
            if mask_image is not None:
                return StableDiffusionInpaintPipeline(
                    **self.pipe.components,
                    requires_safety_checker=False,
                )(
                    callback=callback,
                    generator=generator,
                    guidance_scale=cfg_scale,
                    height=height,
                    image=source_image,
                    mask_image=mask_image,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_images_per_prompt=image_count,
                    num_inference_steps=steps,
                    output_type=output_type,
                    prompt_embeds=prompt_embeds,
                    strength=strength,
                    width=width,
                ).images
            elif source_image is not None:
                if self.base_model_type == BaseModelType.SDXL or self.base_model_type == BaseModelType.SDXL_REFINER:
                    return StableDiffusionXLImg2ImgPipeline(
                        **self.pipe.components,
                        requires_aesthetics_score=self.base_model_type == BaseModelType.SDXL_REFINER,
                    )(
                        callback=callback,
                        denoising_start=denoising_start,
                        denoising_end=denoising_end,
                        generator=generator,
                        guidance_scale=cfg_scale,
                        image=source_image,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_images_per_prompt=image_count,
                        num_inference_steps=steps,
                        output_type=output_type,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        prompt_embeds=prompt_embeds,
                        strength=strength,
                    ).images
                else:
                    return StableDiffusionImg2ImgPipeline(
                        **self.pipe.components,
                        requires_safety_checker=False,
                    )(
                        callback=callback,
                        generator=generator,
                        guidance_scale=cfg_scale,
                        image=source_image,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_images_per_prompt=image_count,
                        num_inference_steps=steps,
                        output_type=output_type,
                        prompt_embeds=prompt_embeds,
                        strength=strength,
                    ).images
            else:
                if self.base_model_type == BaseModelType.SDXL:
                    return self.pipe(
                        callback=callback,
                        denoising_end=denoising_end,
                        generator=generator,
                        guidance_scale=cfg_scale,
                        height=height,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_images_per_prompt=image_count,
                        num_inference_steps=steps,
                        output_type=output_type,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        prompt_embeds=prompt_embeds,
                        width=width,
                    ).images
                else:
                    return self.pipe(
                        callback=callback,
                        generator=generator,
                        guidance_scale=cfg_scale,
                        height=height,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_images_per_prompt=image_count,
                        num_inference_steps=steps,
                        output_type=output_type,
                        prompt_embeds=prompt_embeds,
                        width=width,
                    ).images

    def load(
        self,
        model: str,
        safety_checker: bool,
        control_net: Optional[ControlNetParams],
        base_pipe: Optional[DiffusionPipeline],
    ):
        # ControlNet
        new_control_nets = []
        new_control_net_names = []
        if control_net:
            for condition in control_net.conditions:
                try:
                    index = self.control_net_names.index(condition.model)
                    control_net = self.control_nets[index]
                except ValueError:
                    print("Loading ControlNet", condition.model)
                    model_info = config.models[condition.model]
                    variant = "fp16" if self.torch_dtype == torch.float16 else None

                    if model_info.local:
                        root, _ = os.path.splitext(model_info.path)
                        config_file = f"{root}.yaml"

                        control_net = ControlNetModel.from_single_file(
                            model_info.path,
                            # torch_dtype=self.torch_dtype,     # not working
                            config_file=config_file,
                        )
                    else:
                        control_net = ControlNetModel.from_pretrained(
                            model_info.path,
                            subfolder=model_info.subfolder,
                            torch_dtype=self.torch_dtype,
                            variant=variant,
                        )
                    control_net.to(self.device)
                    control_net.set_attention_slice("auto")

                new_control_nets.append(control_net)
                new_control_net_names.append(condition.model)

        self.control_nets = new_control_nets
        self.control_net_names = new_control_net_names

        # Pipeline
        if self.model != model or self.safety_checker != safety_checker:
            self.unload()
            gc.collect()

        if not self.pipe:
            print("Loading Stable Diffusion Pipeline", model)
            model_info = config.models[model]
            variant = "fp16" if self.torch_dtype == torch.float16 else None

            if model_info.base == BaseModelType.SD_1 or model_info.base == BaseModelType.SD_2:
                if model_info.local:
                    pipe = StableDiffusionPipeline.from_single_file(
                        model_info.path,
                        torch_dtype=self.torch_dtype,
                        load_safety_checker=safety_checker,
                    )
                else:
                    if safety_checker:
                        pipe = StableDiffusionPipeline.from_pretrained(
                            model_info.path,
                            torch_dtype=self.torch_dtype,
                            variant=variant,
                        )
                    else:
                        pipe = StableDiffusionPipeline.from_pretrained(
                            model_info.path,
                            torch_dtype=self.torch_dtype,
                            variant=variant,
                            safety_checker=None,
                            requires_safety_checker=False,
                        )
            elif model_info.base == BaseModelType.SDXL:
                if model_info.local:
                    # TODO - vae setting
                    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        model_info.path,
                        torch_dtype=self.torch_dtype,
                        vae=vae,
                    )
                else:
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        model_info.path,
                        torch_dtype=self.torch_dtype,
                        variant=variant,
                    )
            elif model_info.base == BaseModelType.SDXL_REFINER:
                if model_info.local:
                    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                        model_info.path,
                        torch_dtype=self.torch_dtype,
                        text_encoder_2=base_pipe.text_encoder_2,
                        vae=base_pipe.vae,
                    )
                else:
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_info.path,
                        torch_dtype=self.torch_dtype,
                        variant=variant,
                        text_encoder_2=base_pipe.text_encoder_2,
                        vae=base_pipe.vae,
                    )
            else:
                raise ValueError("Unsupported base model: ", model_info.base)

            pipe.to(self.device)
            pipe.enable_attention_slicing()
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload()
            pipe.backup_weights = {}

            # Textual Inversions
            if isinstance(pipe, TextualInversionLoaderMixin):
                data = [
                    (key, info.path)
                    for key, info in config.models.items()
                    if info.type == "textual-inversion" and info.base == model_info.base
                ]
                if data:
                    tokens, paths = zip(*data)
                    print("Loading Textual Inversions")
                    pipe.load_textual_inversion(list(paths), list(tokens))

            # Compel
            compel2 = None
            if model_info.base == BaseModelType.SD_1 or model_info.base == BaseModelType.SD_2:
                compel = Compel(
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                    textual_inversion_manager=DiffusersTextualInversionManager(pipe),
                )
            elif model_info.base == BaseModelType.SDXL:
                compel = Compel(
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=False,
                )

                compel2 = Compel(
                    tokenizer=pipe.tokenizer_2,
                    text_encoder=pipe.text_encoder_2,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=True,
                )
            elif model_info.base == BaseModelType.SDXL_REFINER:
                compel = Compel(
                    tokenizer=pipe.tokenizer_2,
                    text_encoder=pipe.text_encoder_2,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=True,
                )
            else:
                raise ValueError("Unsupported base model: ", model_info.base)

            self.model = model
            self.base_model_type = model_info.base
            self.safety_checker = safety_checker
            self.pipe = pipe
            self.scheduler_config = pipe.scheduler.config.copy()
            self.compel = compel
            self.compel2 = compel2

    def unload(self):
        self.model = None
        self.base_model_type = None
        self.safety_checker = None
        self.pipe = None
        self.scheduler_config = None
        self.compel = None
        self.compel2 = None
        gc.collect()

    def set_scheduler(self, scheduler: str):
        scheduler_cls, config_params = scheduler_registry.DICT.get(scheduler, (EulerAncestralDiscreteScheduler, {}))
        self.pipe.scheduler = scheduler_cls.from_config({**self.scheduler_config, **config_params})

    def set_loras(self, loras: list[LoraModelParams]):
        if self.base_model_type == BaseModelType.SDXL or self.base_model_type == BaseModelType.SDXL_REFINER:
            # Use diffusers implementation
            self.pipe.unload_lora_weights()
            if loras:
                if len(loras) == 1:
                    lora_weight = loras[0]
                    info = config.models.get(lora_weight.model)
                    if info:
                        self.pipe.load_lora_weights(info.path)
                        self.pipe._lora_scale = lora_weight.weight
                    else:
                        print("Unknown LoRA: ", lora_weight.model)
                else:
                    raise ValueError("Only 1 LoRA currently supported")
        else:
            lora_models = []
            lora_multipliers = []
            for lora_entry in loras:
                info = config.models.get(lora_entry.model)
                if info:
                    lora_models.append(lora.load(info.path, self.device, self.torch_dtype))
                    lora_multipliers.append(lora_entry.weight)
                else:
                    print("Unknown LoRA: ", lora_entry.model)

            lora.apply(self.pipe, lora_models, lora_multipliers)

    def preview(self, latents):
        # Code from InvokeAI
        # https://github.com/invoke-ai/InvokeAI/blob/89b82b3dc4892f2bbf6d15f4e39c56225a54f3a6/invokeai/app/util/step_callback.py#L12

        def to_image(samples, latent_rgb_factors):
            latent_image = samples[0].permute(1, 2, 0) @ latent_rgb_factors
            latents_ubyte = (
                ((latent_image + 1) / 2).clamp(0, 1).mul(0xFF).byte()  # change scale from -1..1 to 0..1  # to 0..255
            ).cpu()

            return Image.fromarray(latents_ubyte.numpy())

        if self.base_model_type in [BaseModelType.SDXL, BaseModelType.SDXL_REFINER]:
            # fast latents preview matrix for sdxl
            # generated by @StAlKeR7779
            sdxl_latent_rgb_factors = torch.tensor(
                [
                    #   R        G        B
                    [0.3816, 0.4930, 0.5320],
                    [-0.3753, 0.1631, 0.1739],
                    [0.1770, 0.3588, -0.2048],
                    [-0.4350, -0.2644, -0.4289],
                ],
                dtype=latents.dtype,
                device=latents.device,
            )

            return to_image(latents, sdxl_latent_rgb_factors)
        else:
            # origingally adapted from code by @erucipe and @keturn here:
            # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7

            # these updated numbers for v1.5 are from @torridgristle
            v1_5_latent_rgb_factors = torch.tensor(
                [
                    #    R        G        B
                    [0.3444, 0.1385, 0.0670],  # L1
                    [0.1247, 0.4027, 0.1494],  # L2
                    [-0.3192, 0.2513, 0.2103],  # L3
                    [-0.1307, -0.1874, -0.7445],  # L4
                ],
                dtype=latents.dtype,
                device=latents.device,
            )

            return to_image(latents, v1_5_latent_rgb_factors)
