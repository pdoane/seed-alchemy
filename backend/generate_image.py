import io
import json
import os
from typing import Optional

import torch
from PIL import Image, ImageOps, PngImagePlugin

from . import config, messages, utils
from .control_net import ControlNetProcessor
from .device import default_device, default_dtype
from .esrgan import ESRGANProcessor
from .gfpgan import GFPGANProcessor
from .models import ImageRequest, PreviewType, ProcessRequest
from .session import CancelException, Session
from .tiny_vae import TinyVAE
from .universal_pipeline import UniversalPipeline


def align_down(n: int, align: int) -> int:
    return align * (n // align)


class ImageGenerator:
    def __init__(self, controlnet_processor: ControlNetProcessor):
        self.device = default_device()
        self.torch_dtype = default_dtype()

        self.base_pipeline = UniversalPipeline()
        self.refiner_pipeline = UniversalPipeline()
        self.esrgan = ESRGANProcessor()
        self.gfpgan = GFPGANProcessor()
        self.controlnet_processor = controlnet_processor
        self.tiny_vae = TinyVAE()

        # Generation state
        self.req = None
        self.session = None
        self.step = -1

    def __call__(self, req: ImageRequest, session: Optional[Session]):
        # Init
        self.req = req
        self.session = session
        self.step = 0

        # Source image
        source_image = None
        if req.img2img:
            full_path = config.get_image_path(req.user, req.img2img.source)
            with Image.open(full_path) as image:
                image = image.convert("RGB")
                source_image = image.copy()

        # Mask image
        mask_image = None
        if req.inpaint:
            full_path = config.get_image_path(req.user, req.inpaint.source)
            with Image.open(full_path) as image:
                if req.inpaint.use_alpha_channel:
                    image = image.split()[3].convert("L")
                else:
                    image = image.convert("L")
                if req.inpaint.invert_mask:
                    image = ImageOps.invert(image)

                image = image.resize((req.width, req.height), Image.Resampling.LANCZOS)
                mask_image = image.copy()

        # Conditioning images
        control_images = []
        if req.control_net:
            for condition in req.control_net.conditions:
                full_path = config.get_image_path(req.user, condition.source)
                with Image.open(full_path) as image:
                    if condition.processor != "none":
                        image = self.controlnet_processor(
                            image, min(req.width, req.height), condition.processor, condition.params
                        )
                        image = image.resize((req.width, req.height), Image.Resampling.LANCZOS)
                    control_images.append(image.copy())

        # Pipelines
        self.base_pipeline.load(req.model, req.safety_checker, req.control_net, None)
        self.base_pipeline.set_scheduler(req.scheduler)
        self.base_pipeline.set_loras(req.lora.entries if req.lora else [])

        if req.refiner:
            self.refiner_pipeline.load(req.refiner.model, req.safety_checker, None, self.base_pipeline.pipe)
            self.refiner_pipeline.set_scheduler(req.scheduler)
        else:
            self.refiner_pipeline.unload()

        # Seed
        generator = torch.Generator().manual_seed(req.seed)

        try:
            # Generate
            if req.img2img and req.img2img.noise == 0.0:
                images = [source_image] * req.image_count
            else:
                if source_image is not None:
                    source_image = source_image.resize((req.width, req.height), Image.Resampling.LANCZOS)

                images = self.base_pipeline(
                    image_count=req.image_count,
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    steps=req.steps,
                    denoising_start=None,
                    denoising_end=req.refiner.high_noise_end if req.refiner else None,
                    cfg_scale=req.cfg_scale,
                    width=req.width,
                    height=req.height,
                    generator=generator,
                    noise=req.img2img.noise if req.img2img else None,
                    source_image=source_image,
                    mask_image=mask_image,
                    control_net=req.control_net,
                    control_images=control_images,
                    output_type="latent" if req.refiner else "pil",
                    callback=self.callback,
                )

            # Post-process
            output_paths = []
            for image in images:
                # Refiner
                if req.refiner:
                    image = self.refiner_pipeline(
                        image_count=1,
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt,
                        steps=req.steps if req.refiner.high_noise_end is not None else req.refiner.steps,
                        denoising_start=req.refiner.high_noise_end,
                        denoising_end=None,
                        cfg_scale=req.refiner.cfg_scale,
                        width=req.width,
                        height=req.height,
                        generator=generator,
                        noise=req.refiner.noise if req.refiner.high_noise_end is None else None,
                        source_image=image,
                        mask_image=mask_image,
                        control_net=None,
                        control_images=None,
                        output_type="pil",
                        callback=self.callback,
                    )[0]

                # High Resolution
                if req.high_res:
                    high_res_width = align_down(int(req.width * req.high_res.factor), 8)
                    high_res_height = align_down(int(req.height * req.high_res.factor), 8)
                    source_image = image.resize((high_res_width, high_res_height), Image.Resampling.LANCZOS)
                    if mask_image is not None:
                        mask_image = mask_image.resize((high_res_width, high_res_height), Image.Resampling.LANCZOS)

                    orig_control_images = control_images
                    control_images = []
                    for control_image in orig_control_images:
                        control_images.append(
                            control_image.resize((high_res_width, high_res_height), Image.Resampling.LANCZOS)
                        )

                    image = self.base_pipeline(
                        image_count=1,
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt,
                        steps=req.high_res.steps,
                        denoising_start=None,
                        denoising_end=None,
                        cfg_scale=req.high_res.cfg_scale,
                        width=high_res_width,
                        height=high_res_height,
                        generator=generator,
                        noise=req.high_res.noise,
                        source_image=source_image,
                        mask_image=mask_image,
                        control_net=req.control_net,
                        control_images=control_images,
                        output_type="pil",
                        callback=self.callback,
                    )[0]

                # ESRGAN
                if req.upscale:
                    upscaled_image = self.esrgan(
                        image=image,
                        upscale_factor=req.upscale.factor,
                        denoising_strength=req.upscale.denoising,
                        blend_strength=req.upscale.blend,
                        float32=True,  # TODO - 16bit
                    )
                    self.next_step()
                else:
                    upscaled_image = image

                # GFPGAN
                if req.face:
                    image = self.gfpgan(
                        image=image,
                        upscale_factor=req.upscale.factor if req.upscale else 1,
                        upscaled_image=upscaled_image,
                        blend_strength=req.face.blend,
                    )
                    self.next_step()
                else:
                    image = upscaled_image

                # Metadata
                filtered_dict = utils.remove_none_fields(req.dict())
                for key in ["session_id", "generator_id", "user", "collection", "image_count", "preview"]:
                    if key in filtered_dict:
                        filtered_dict.pop(key)
                png_info = PngImagePlugin.PngInfo()
                png_info.add_text("seed-alchemy", json.dumps(filtered_dict))

                # Serialize
                output_path = config.generate_output_path(req.user, req.collection)
                full_path = config.get_image_path(req.user, output_path)
                with open(full_path, "wb") as f:
                    image.save(f, pnginfo=png_info)
                    f.flush()
                    os.fsync(f.fileno())

                self.next_step()

                output_paths.append(utils.normalize_path(output_path))
            return output_paths

        except CancelException:
            return []

    def callback(self, step: int, timestep: int, latents: torch.FloatTensor):
        req = self.req
        self.next_step()

        if self.session:
            if req.high_res:
                preview_width = align_down(int(req.width * req.high_res.factor), 8)
                preview_height = align_down(int(req.height * req.high_res.factor), 8)
            else:
                preview_width = req.width
                preview_height = req.height

            if req.upscale:
                preview_width *= req.upscale.factor
                preview_height *= req.upscale.factor

            if req.preview == PreviewType.TINY_VAE:
                self.tiny_vae.load(self.base_pipeline.base_model_type)
                image = self.tiny_vae.decode(latents)
                image = image.resize((preview_width, preview_height), Image.BILINEAR)
            else:
                self.tiny_vae.unload()
                image = self.base_pipeline.preview(latents)
                image = image.resize((preview_width, preview_height), Image.NEAREST)

            buffered = io.BytesIO()
            image.save(buffered, format="png")

            self.session.queue.sync_q.put(messages.build_image(req.generator_id, buffered.getvalue()))

    def next_step(self):
        req = self.req
        self.step += 1

        if self.session:
            if self.session.cancel:
                self.session.cancel = False
                raise CancelException()

            steps = self.compute_steps()
            progress_amount = int(self.step * 100 / steps)
            self.session.queue.sync_q.put(messages.build_progress(req.generator_id, progress_amount))

    def compute_steps(self):
        req = self.req

        steps_per_image = 1
        if req.refiner and req.refiner.high_noise_end is None:
            steps_per_image += int(req.refiner.steps * req.refiner.noise)
        if req.high_res:
            steps_per_image += int(req.high_res.steps * req.high_res.noise)
        if req.upscale:
            steps_per_image += 1
        if req.face:
            steps_per_image += 1

        if req.img2img:
            pipeline_steps = int(req.steps * req.img2img.noise)
        else:
            pipeline_steps = req.steps

        return pipeline_steps + req.image_count * steps_per_image


class PreviewProcessor:
    def __init__(self, controlnet_processor: ControlNetProcessor):
        self.controlnet_processor = controlnet_processor

    def __call__(self, req: ProcessRequest):
        full_path = config.get_image_path(req.user, req.source)
        with Image.open(full_path) as image:
            image = self.controlnet_processor(image, 512, req.processor, req.params)

            output_path = config.generate_output_path(req.user, req.collection)
            full_path = config.get_image_path(req.user, output_path)
            with open(full_path, "wb") as f:
                image.save(f)
                f.flush()
                os.fsync(f.fileno())

        return utils.normalize_path(output_path)
