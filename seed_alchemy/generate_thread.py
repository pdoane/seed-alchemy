import os

import torch
from compel import Compel, PromptParser
from compel.diffusers_textual_inversion_manager import DiffusersTextualInversionManager
from PIL import Image, ImageOps, PngImagePlugin
from PySide6.QtCore import Signal

from . import configuration, control_net_config, scheduler_registry, utils
from .backend import BackendTask, CancelTaskException
from .image_metadata import ControlNetMetadata, ImageMetadata, Img2ImgMetadata
from .pipelines import GenerateRequest, ImagePipeline, PipelineCache
from .processors import ESRGANProcessor, GFPGANProcessor, ProcessorBase

pipeline_cache: PipelineCache = PipelineCache()
generate_preprocessor: ProcessorBase = None


def latents_to_pil(latents: torch.FloatTensor):
    # Code from InvokeAI
    # https://github.com/invoke-ai/InvokeAI/blob/a1cd4834d127641a865438e668c5c7f050e83587/invokeai/backend/generator/base.py#L502

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

    latent_image = latents[0].permute(1, 2, 0) @ v1_5_latent_rgb_factors
    latents_ubyte = (
        ((latent_image + 1) / 2).clamp(0, 1).mul(0xFF).byte()  # change scale from -1..1 to 0..1  # to 0..255
    ).cpu()

    return Image.fromarray(latents_ubyte.numpy())


def align_down(n: int, align: int) -> int:
    return align * (n // align)


class GenerateImageTask(BackendTask):
    task_progress = Signal(int)
    image_preview = Signal(Image.Image)
    image_complete = Signal(str)

    def __init__(self, req: GenerateRequest):
        super().__init__()

        self.cancel = False
        self.step = 0
        self.req = req
        self.req.callback = self.generate_callback

    def run_(self):
        global generate_preprocessor

        # load pipeline
        pipeline = ImagePipeline(pipeline_cache, self.req.image_metadata)
        pipeline_cache.pipeline = pipeline
        pipe = pipeline.pipe

        # Source image
        img2img_meta = self.req.image_metadata.img2img
        if img2img_meta:
            full_path = os.path.join(configuration.IMAGES_PATH, img2img_meta.source)
            with Image.open(full_path) as image:
                image = image.convert("RGB")
                # Delay resize until generation for enhance operations
                self.req.source_image = image.copy()

        # Mask image
        inpaint_meta = self.req.image_metadata.inpaint

        if inpaint_meta:
            full_path = os.path.join(configuration.IMAGES_PATH, inpaint_meta.source)
            with Image.open(full_path) as image:
                if inpaint_meta.use_alpha_channel:
                    image = image.split()[3].convert("L")
                else:
                    image = image.convert("L")
                if inpaint_meta.invert_mask:
                    image = ImageOps.invert(image)

                image = image.resize(
                    (self.req.image_metadata.width, self.req.image_metadata.height), Image.Resampling.LANCZOS
                )
                self.req.mask_image = image.copy()

        # Conditioning images
        control_net_meta = self.req.image_metadata.control_net
        if control_net_meta:
            for condition_meta in control_net_meta.conditions:
                source_path = condition_meta.source
                full_path = os.path.join(configuration.IMAGES_PATH, source_path)
                with Image.open(full_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(
                        (self.req.image_metadata.width, self.req.image_metadata.height), Image.Resampling.LANCZOS
                    )
                    controlnet_conditioning_image = image.copy()

                if condition_meta.preprocessor is not None:
                    preprocessor_type = control_net_config.preprocessors.get(condition_meta.preprocessor)
                    if preprocessor_type:
                        if not isinstance(generate_preprocessor, preprocessor_type):
                            generate_preprocessor = preprocessor_type(configuration.torch_device)
                        controlnet_conditioning_image = generate_preprocessor(
                            controlnet_conditioning_image, condition_meta.params
                        )
                        if self.req.reduce_memory:
                            generate_preprocessor = None

                self.req.control_images.append(controlnet_conditioning_image)

        # scheduler
        scheduler_cls, config_params = scheduler_registry.DICT.get(
            self.req.image_metadata.scheduler, scheduler_registry.DICT["euler_a"]
        )
        pipe.scheduler = scheduler_cls.from_config({**pipeline.scheduler_config, **config_params})

        # generator
        self.req.generator = torch.Generator().manual_seed(self.req.image_metadata.seed)

        # prompt parsing
        pp = PromptParser()
        conjunction = pp.parse_conjunction(self.req.image_metadata.prompt)

        # loras
        loras = []
        for lora_weight in conjunction.lora_weights:
            loras.append((lora_weight.model, lora_weight.weight))
        pipeline.set_loras(loras)

        # prompt weighting
        textual_inversion_manager = DiffusersTextualInversionManager(pipe)
        compel_proc = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            textual_inversion_manager=textual_inversion_manager,
        )
        self.req.prompt_embeds = compel_proc(self.req.image_metadata.prompt)
        self.req.negative_prompt_embeds = compel_proc(self.req.image_metadata.negative_prompt)

        # generate
        if img2img_meta and img2img_meta.noise == 0.0:
            images = [self.req.source_image] * self.req.num_images_per_prompt
        else:
            if self.req.source_image is not None:
                self.req.source_image = self.req.source_image.resize(
                    (self.req.image_metadata.width, self.req.image_metadata.height), Image.Resampling.LANCZOS
                )
            images = pipeline(self.req)

        for image in images:
            loop_count = 2 if self.req.image_metadata.high_res else 1

            for i in range(loop_count):
                # ESRGAN
                upscale_meta = self.req.image_metadata.upscale
                if upscale_meta:
                    esrgan = ESRGANProcessor(configuration.torch_device)
                    esrgan.upscale_factor = upscale_meta.factor
                    esrgan.denoising_strength = upscale_meta.denoising
                    esrgan.blend_strength = upscale_meta.blend
                    upscaled_image = esrgan(image, [])
                    self.next_step()
                else:
                    upscaled_image = image

                # GFPGAN
                face_meta = self.req.image_metadata.face
                if face_meta:
                    gfpgan = GFPGANProcessor(configuration.torch_device)
                    gfpgan.upscale_factor = upscale_meta.factor if upscale_meta else 1
                    gfpgan.upscaled_image = upscaled_image
                    gfpgan.blend_strength = face_meta.blend
                    image = gfpgan(image, [])
                    self.next_step()
                else:
                    image = upscaled_image

                # High Resolution
                high_res_meta = self.req.image_metadata.high_res
                if i == 0 and high_res_meta:
                    high_res_width = align_down(int(self.req.image_metadata.width * high_res_meta.factor), 8)
                    high_res_height = align_down(int(self.req.image_metadata.height * high_res_meta.factor), 8)
                    source_image = image.resize((high_res_width, high_res_height), Image.Resampling.LANCZOS)

                    high_res_req = GenerateRequest()
                    high_res_req.image_metadata = ImageMetadata()
                    high_res_req.source_image = source_image
                    high_res_req.image_metadata.num_inference_steps = high_res_meta.steps
                    high_res_req.image_metadata.guidance_scale = high_res_meta.guidance_scale
                    high_res_req.image_metadata.width = high_res_width
                    high_res_req.image_metadata.height = high_res_height
                    high_res_req.image_metadata.img2img = Img2ImgMetadata(noise=high_res_meta.noise)
                    high_res_req.num_images_per_prompt = 1
                    high_res_req.generator = torch.Generator().manual_seed(self.req.image_metadata.seed)
                    high_res_req.prompt_embeds = self.req.prompt_embeds
                    high_res_req.negative_prompt_embeds = self.req.negative_prompt_embeds
                    high_res_req.callback = self.req.callback

                    control_net_meta = self.req.image_metadata.control_net
                    if control_net_meta:
                        high_res_req.control_images = []
                        for controlnet_conditioning_image in self.req.control_images:
                            controlnet_conditioning_image = controlnet_conditioning_image.resize(
                                (high_res_width, high_res_height), Image.Resampling.LANCZOS
                            )
                            high_res_req.control_images.append(controlnet_conditioning_image)

                        high_res_req.image_metadata.control_net = ControlNetMetadata(
                            guidance_start=control_net_meta.guidance_start,
                            guidance_end=control_net_meta.guidance_end,
                            conditions=control_net_meta.conditions,
                        )

                    image = pipeline(high_res_req)[0]

            # Output
            collection = self.req.collection
            png_info = PngImagePlugin.PngInfo()
            self.req.image_metadata.save_to_png_info(png_info)

            def io_operation():
                next_image_id = utils.next_image_id(os.path.join(configuration.IMAGES_PATH, collection))
                output_path = os.path.join(collection, "{:05d}.png".format(next_image_id))
                full_path = os.path.join(configuration.IMAGES_PATH, output_path)

                image.save(full_path, pnginfo=png_info)
                return output_path

            output_path = utils.retry_on_failure(io_operation)

            self.next_step()

            self.image_complete.emit(output_path)

    def generate_callback(self, step: int, timestep: int, latents: torch.FloatTensor):
        self.next_step()

        high_res_meta = self.req.image_metadata.high_res
        if high_res_meta:
            preview_width = align_down(int(self.req.image_metadata.width * high_res_meta.factor), 8)
            preview_height = align_down(int(self.req.image_metadata.height * high_res_meta.factor), 8)
        else:
            preview_width = self.req.image_metadata.width
            preview_height = self.req.image_metadata.height

        upscale_meta = self.req.image_metadata.upscale
        if upscale_meta:
            preview_width *= upscale_meta.factor
            preview_height *= upscale_meta.factor

        pil_image = latents_to_pil(latents)
        pil_image = pil_image.resize((preview_width, preview_height), Image.NEAREST)
        self.image_preview.emit(pil_image)

    def next_step(self):
        if self.cancel:
            raise CancelTaskException()

        self.step += 1
        steps = self.compute_total_steps()
        progress_amount = self.step * 100 / steps
        self.task_progress.emit(progress_amount)

    def compute_total_steps(self):
        loop_count = 2 if self.req.image_metadata.high_res else 1

        steps_per_image = 1
        if self.req.image_metadata.upscale:
            steps_per_image += loop_count
        if self.req.image_metadata.face:
            steps_per_image += loop_count
        high_res_meta = self.req.image_metadata.high_res
        if high_res_meta:
            steps_per_image += int(high_res_meta.steps * high_res_meta.noise)

        pipeline_steps = self.req.image_metadata.num_inference_steps
        img2img_meta = self.req.image_metadata.img2img
        if img2img_meta:
            pipeline_steps = int(pipeline_steps * img2img_meta.noise)

        return pipeline_steps + self.req.num_images_per_prompt * steps_per_image
