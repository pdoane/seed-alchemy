import gc
import os
from typing import Callable, Optional, Union

import torch
from compel import Compel, ReturnedEmbeddingsType, SplitLongTextMode
from compel.diffusers_textual_inversion_manager import DiffusersTextualInversionManager
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxTransformer2DModel,
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
        negative_prompt: str | None,
        steps: int,
        denoising_start: Optional[float],
        denoising_end: Optional[float],
        cfg_scale: float,
        width: int,
        height: int,
        generator: torch.Generator,
        noise: Optional[float],
        clip_skip: Optional[int],
        source_image: Optional[Union[Image.Image, torch.FloatTensor]],
        mask_image: Optional[Union[Image.Image, torch.FloatTensor]],
        control_net: Optional[ControlNetParams],
        control_images: Optional[list[Union[Image.Image, torch.FloatTensor]]],
        output_type: str,
        callback_on_step_end: Optional[Callable[[int, int, torch.FloatTensor], None]],
    ):
        self.width = width
        self.height = height

        # Step adjustment
        # if source_image is not None and noise is not None:
        #     diffusers_steps = int(steps * noise)
        #     scaled_steps = int(diffusers_steps / noise)
        #     while steps != int(scaled_steps * noise):
        #         scaled_steps += 1
        #     steps = scaled_steps

        # Prompt
        if self.base_model_type in [BaseModelType.SD_1, BaseModelType.SD_2]:
            prompt_embeds = self.compel(prompt)
            if negative_prompt is not None:
                negative_prompt_embeds = self.compel(negative_prompt)
                [prompt_embeds, negative_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])

            else:
                negative_prompt_embeds = None

        elif self.base_model_type == BaseModelType.SDXL:
            prompt2 = prompt
            negative_prompt2 = negative_prompt

            prompt1_embeds = self.compel(prompt)
            prompt2_embeds, pooled_prompt_embeds = self.compel2(prompt2)

            if negative_prompt is not None:
                negative_prompt1_embeds = self.compel(negative_prompt)
                negative_prompt2_embeds, negative_pooled_prompt_embeds = self.compel2(negative_prompt2)
                [prompt1_embeds, negative_prompt1_embeds] = self.compel.pad_conditioning_tensors_to_same_length([prompt1_embeds, negative_prompt1_embeds])
                [prompt2_embeds, negative_prompt2_embeds] = self.compel2.pad_conditioning_tensors_to_same_length([prompt2_embeds, negative_prompt2_embeds])
                # [pooled_prompt_embeds, negative_pooled_prompt_embeds] = self.compel2.pad_conditioning_tensors_to_same_length([pooled_prompt_embeds, negative_pooled_prompt_embeds])
                prompt_embeds = torch.cat((prompt1_embeds, prompt2_embeds), dim=-1)
                negative_prompt_embeds = torch.cat((negative_prompt1_embeds, negative_prompt2_embeds), dim=-1)
            else:
                prompt_embeds = torch.cat((prompt1_embeds, prompt2_embeds), dim=-1)
                negative_prompt_embeds = None
                negative_pooled_prompt_embeds = None

        elif self.base_model_type == BaseModelType.SDXL_REFINER:
            prompt_embeds, pooled_prompt_embeds = self.compel(prompt)
            if negative_prompt is not None:
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.compel(negative_prompt)
                [prompt_embeds, negative_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
                [pooled_prompt_embeds, negative_pooled_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length([pooled_prompt_embeds, negative_pooled_prompt_embeds])
            else:
                negative_prompt_embeds = None
                negative_pooled_prompt_embeds = None

        elif self.base_model_type == BaseModelType.FLUX:
            # TODO - expose 2nd prompt
            prompt2 = prompt
            negative_prompt2 = negative_prompt

        else:
            raise ValueError("Unsupported base model: ", self.base_model_type)

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
                if self.base_model_type in [BaseModelType.SD_1, BaseModelType.SD_2]:
                    return StableDiffusionControlNetInpaintPipeline(
                        **self.pipe.components,
                        controlnet=controlnet,
                        requires_safety_checker=False,
                    )(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                else:
                    raise ValueError("Unsupported base model: ", self.base_model_type)
            elif source_image is not None:
                if self.base_model_type in [BaseModelType.SD_1, BaseModelType.SD_2]:
                    return StableDiffusionControlNetImg2ImgPipeline(
                        **self.pipe.components,
                        controlnet=controlnet,
                        requires_safety_checker=False,
                    )(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                    raise ValueError("Unsupported base model: ", self.base_model_type)
            else:
                if self.base_model_type in [BaseModelType.SD_1, BaseModelType.SD_2]:
                    return StableDiffusionControlNetPipeline(
                        **self.pipe.components,
                        controlnet=controlnet,
                        requires_safety_checker=False,
                    )(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                elif self.base_model_type == BaseModelType.SDXL:
                    return StableDiffusionXLControlNetPipeline(
                        **self.pipe.components,
                        controlnet=controlnet,
                    )(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                    raise ValueError("Unsupported base model: ", self.base_model_type)
        else:
            if mask_image is not None:
                if self.base_model_type in [BaseModelType.SD_1, BaseModelType.SD_2]:
                    return StableDiffusionInpaintPipeline(
                        **self.pipe.components,
                        requires_safety_checker=False,
                    )(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                else:
                    raise ValueError("Unsupported base model: ", self.base_model_type)
            elif source_image is not None:
                if self.base_model_type in [BaseModelType.SD_1, BaseModelType.SD_2]:
                    return StableDiffusionImg2ImgPipeline(
                        **self.pipe.components,
                        requires_safety_checker=False,
                    )(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                elif self.base_model_type in [BaseModelType.SDXL, BaseModelType.SDXL_REFINER]:
                    return StableDiffusionXLImg2ImgPipeline(
                        **self.pipe.components,
                        requires_aesthetics_score=self.base_model_type == BaseModelType.SDXL_REFINER,
                    )(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                elif self.base_model_type == BaseModelType.FLUX:
                    return FluxImg2ImgPipeline(
                        scheduler=self.pipe.scheduler,
                        text_encoder_2=self.pipe.text_encoder_2,
                        text_encoder=self.pipe.text_encoder,
                        tokenizer_2=self.pipe.tokenizer_2,
                        tokenizer=self.pipe.tokenizer,
                        transformer=self.pipe.transformer,
                        vae=self.pipe.vae,
                    )(
                        callback_on_step_end=callback_on_step_end,
                        generator=generator,
                        guidance_scale=cfg_scale,
                        height=height,
                        image=source_image,
                        num_images_per_prompt=image_count,
                        num_inference_steps=steps,
                        output_type=output_type,
                        prompt=prompt,
                        prompt_2=prompt2,
                        strength=strength,
                        width=width,
                    ).images
                else:
                    raise ValueError("Unsupported base model: ", self.base_model_type)
            else:
                if self.base_model_type in [BaseModelType.SD_1, BaseModelType.SD_2]:
                    return self.pipe(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                elif self.base_model_type == BaseModelType.SDXL:
                    return self.pipe(
                        callback_on_step_end=callback_on_step_end,
                        clip_skip=clip_skip,
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
                elif self.base_model_type == BaseModelType.FLUX:
                    return self.pipe(
                        callback_on_step_end=callback_on_step_end,
                        generator=generator,
                        guidance_scale=cfg_scale,
                        height=height,
                        negative_prompt=negative_prompt,
                        negative_prompt_2=negative_prompt2,
                        num_images_per_prompt=image_count,
                        num_inference_steps=steps,
                        output_type=output_type,
                        prompt=prompt,
                        prompt_2=prompt2,
                        width=width,
                    ).images
                else:
                    raise ValueError("Unsupported base model: ", self.base_model_type)

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
            print("Loading Pipeline", model)
            model_info = config.models[model]
            variant = "fp16" if self.torch_dtype == torch.float16 else None

            if model_info.base == BaseModelType.SD_1 or model_info.base == BaseModelType.SD_2:
                if model_info.local:
                    if safety_checker:
                        pipe = StableDiffusionPipeline.from_single_file(
                            model_info.path,
                            torch_dtype=self.torch_dtype,
                        )
                    else:
                        pipe = StableDiffusionPipeline.from_single_file(
                            model_info.path,
                            torch_dtype=self.torch_dtype,
                            safety_checker=None,
                            requires_safety_checker=False,
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
            elif model_info.base == BaseModelType.FLUX:
                if model_info.local:
                    transformer = FluxTransformer2DModel.from_single_file(model_info.path, torch_dtype=torch.bfloat16)
                    pipe = FluxPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        transformer=transformer,
                        torch_dtype=torch.bfloat16,
                        variant=variant,
                        # pipe = FluxPipeline.from_single_file(
                        #    model_info.path,
                        #    torch_dtype=torch.bfloat16,
                    )
                else:
                    pipe = FluxPipeline.from_pretrained(
                        model_info.path,
                        torch_dtype=torch.bfloat16,
                        variant=variant,
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
                data = [(key, info.path) for key, info in config.models.items() if info.type == "textual-inversion" and info.base == model_info.base]
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
                    truncate_long_prompts=False,
                )
            elif model_info.base == BaseModelType.SDXL:
                compel = Compel(
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=False,
                    truncate_long_prompts=False,
                )

                compel2 = Compel(
                    tokenizer=pipe.tokenizer_2,
                    text_encoder=pipe.text_encoder_2,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=True,
                    truncate_long_prompts=False,
                )
            elif model_info.base == BaseModelType.SDXL_REFINER:
                compel = Compel(
                    tokenizer=pipe.tokenizer_2,
                    text_encoder=pipe.text_encoder_2,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=True,
                    truncate_long_prompts=False,
                )
            elif model_info.base == BaseModelType.FLUX:
                compel = None
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
        if self.base_model_type != BaseModelType.FLUX:
            scheduler_cls, config_params = scheduler_registry.DICT.get(scheduler, (EulerAncestralDiscreteScheduler, {}))
            self.pipe.scheduler = scheduler_cls.from_config({**self.scheduler_config, **config_params})

    def set_loras(self, loras: list[LoraModelParams]):
        if self.base_model_type in [BaseModelType.SDXL, BaseModelType.SDXL_REFINER, BaseModelType.FLUX]:
            # Use diffusers implementation
            self.pipe.unload_lora_weights()
            adapter_names = []
            adapter_weights = []
            n = 0
            for lora_entry in loras:
                info = config.models.get(lora_entry.model)
                if info:
                    adapter_name = f"{n}"
                    self.pipe.load_lora_weights(info.path, adapter_name=adapter_name)
                    adapter_names.append(adapter_name)
                    adapter_weights.append(lora_entry.weight)
                    n += 1
                else:
                    print("Unknown LoRA: ", lora_entry.model)
            if n > 0:
                self.pipe.set_adapters(adapter_names, adapter_weights)

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

            # Clear LoRA model tensors after application to free memory
            for model in lora_models:
                model.layer_elems.clear()
            lora_models.clear()

    def preview(self, latents):
        # Code from InvokeAI
        # https://github.com/invoke-ai/InvokeAI/blob/89b82b3dc4892f2bbf6d15f4e39c56225a54f3a6/invokeai/app/util/step_callback.py#L12

        def to_image(samples, latent_rgb_factors):
            latent_image = samples[0].permute(1, 2, 0) @ latent_rgb_factors
            latents_ubyte = (((latent_image + 1) / 2).clamp(0, 1).mul(0xFF).byte()).cpu()  # change scale from -1..1 to 0..1  # to 0..255

            return Image.fromarray(latents_ubyte.numpy())

        if self.base_model_type in [BaseModelType.SD_1, BaseModelType.SD_2]:
            # originally adapted from code by @erucipe and @keturn here:
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

        elif self.base_model_type in [BaseModelType.SDXL, BaseModelType.SDXL_REFINER]:
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

        elif self.base_model_type in [BaseModelType.FLUX]:

            def _unpack_latents(latents, height, width, vae_scale_factor):
                batch_size, num_patches, channels = latents.shape

                height = 2 * (int(height) // (vae_scale_factor * 2))
                width = 2 * (int(width) // (vae_scale_factor * 2))

                latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
                latents = latents.permute(0, 3, 1, 4, 2, 5)

                latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

                return latents

            latents = _unpack_latents(latents, self.height, self.width, vae_scale_factor=8)

            latents_scaling_factor = 0.3611
            latents_shift_factor = 0.1159
            latents = (latents / latents_scaling_factor) + latents_shift_factor

            W = [
                [-0.01569846272468567, 0.002361269434913993, 0.021068623289465904],
                [0.02266843616962433, 0.026353996247053146, 0.03208939731121063],
                [0.017583753913640976, -0.014552677981555462, -0.00712566077709198],
                [-0.0047706011682748795, -0.003910496830940247, 0.012125629000365734],
                [0.0077856541611254215, 0.005959618370980024, -0.008156429044902325],
                [-0.006002367474138737, 0.0014154225355014205, -0.001960920402780175],
                [0.011962509714066982, 0.02873299829661846, 0.03133808448910713],
                [-0.005278797820210457, -0.018325896933674812, -0.028332434594631195],
                [-0.013505342416465282, -0.003528749104589224, 0.021339554339647293],
                [0.03610225394368172, 0.021061204373836517, -0.01955372467637062],
                [0.0013291906798258424, 0.014273520559072495, 0.006561569403856993],
                [0.020236913114786148, 0.0008685103384777904, 0.006297718733549118],
                [0.018445534631609917, 0.01760716550052166, 0.022904666140675545],
                [-0.033055998384952545, -0.0009437335538677871, -0.024412795901298523],
                [-0.008665954694151878, -0.019846158102154732, -0.006662135943770409],
                [-0.024940500035881996, -0.01634855754673481, -0.008642114698886871],
            ]

            W_tensor = torch.tensor(W, dtype=latents.dtype, device=latents.device)

            latents = latents[0]
            latents = latents.permute(1, 2, 0)
            latents = torch.matmul(latents, W_tensor)
            latents = (((latents + 1) / 2).clamp(0, 1).mul(0xFF).byte()).cpu()
            return Image.fromarray(latents.numpy())
        else:
            raise ValueError("Unsupported base model: ", self.base_model_type)
