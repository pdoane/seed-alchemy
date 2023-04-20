import gc
import os
import traceback

import configuration
import torch
import utils
from compel import Compel
from PIL import Image, PngImagePlugin
from pipelines import GenerateRequest, ImagePipeline, PipelineCache
from processors import ESRGANProcessor, GFPGANProcessor, ProcessorBase
from PySide6.QtCore import QSettings, QThread, Signal

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
            [ 0.3444,  0.1385,  0.0670],  # L1
            [ 0.1247,  0.4027,  0.1494],  # L2
            [-0.3192,  0.2513,  0.2103],  # L3
            [-0.1307, -0.1874, -0.7445],  # L4
        ],
        dtype=latents.dtype,
        device=latents.device,
    )

    latent_image = latents[0].permute(1, 2, 0) @ v1_5_latent_rgb_factors
    latents_ubyte = (
        ((latent_image + 1) / 2)
        .clamp(0, 1)  # change scale from -1..1 to 0..1
        .mul(0xFF)  # to 0..255
        .byte()
    ).cpu()

    return Image.fromarray(latents_ubyte.numpy())

class CancelThreadException(Exception):
    pass

class GenerateThread(QThread):
    task_progress = Signal(int)
    image_preview = Signal(Image.Image)
    image_complete = Signal(str)
    task_complete = Signal()

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)

        self.cancel = False
        self.collection = settings.value('collection')
        self.reduce_memory = settings.value('reduce_memory', type=bool)
        self.req = GenerateRequest()
        self.req.image_metadata.load_from_settings(settings)
        self.req.num_images_per_prompt = int(settings.value('num_images_per_prompt', 1))
        self.req.callback = self.generate_callback
    
    def run(self):
        try:
            self.run_()
        except CancelThreadException:
            pass
        except Exception as e:
            traceback.print_exc()
        
        self.task_complete.emit()
        gc.collect()
 
    def run_(self):
        global generate_preprocessor

        # load pipeline
        pipeline = ImagePipeline(pipeline_cache, self.req.image_metadata)
        pipeline_cache.pipeline = pipeline
        pipe = pipeline.pipe

        # Source image
        if self.req.image_metadata.img2img_enabled:
            source_path = self.req.image_metadata.source_images[self.req.image_metadata.img2img_source]
            full_path = os.path.join(configuration.IMAGES_PATH, source_path)
            with Image.open(full_path) as image:
                image = image.convert('RGB')
                image = image.resize((self.req.image_metadata.width, self.req.image_metadata.height))
                self.req.source_image = image.copy()

        # Conditioning images
        if self.req.image_metadata.control_net_enabled:
            for control_net_meta in self.req.image_metadata.control_nets:
                source_path = self.req.image_metadata.source_images[control_net_meta.image_source]
                full_path = os.path.join(configuration.IMAGES_PATH, source_path)
                with Image.open(full_path) as image:
                    image = image.convert('RGB')
                    image = image.resize((self.req.image_metadata.width, self.req.image_metadata.height))
                    controlnet_conditioning_image = image.copy()

                if control_net_meta.preprocess:
                    control_net_config = configuration.control_net_models[control_net_meta.name]
                    if control_net_config:
                        preprocessor_type = control_net_config.preprocessor
                        if preprocessor_type:
                            if not isinstance(generate_preprocessor, preprocessor_type):
                                generate_preprocessor = preprocessor_type()
                            controlnet_conditioning_image = generate_preprocessor(controlnet_conditioning_image)
                            if self.reduce_memory:
                                generate_preprocessor = None
                
                self.req.controlnet_conditioning_images.append(controlnet_conditioning_image)

        # scheduler
        pipe.scheduler = configuration.schedulers[self.req.image_metadata.scheduler].from_config(pipe.scheduler.config)

        # generator
        self.req.generator = torch.Generator().manual_seed(self.req.image_metadata.seed)

        # prompt weighting
        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        self.req.prompt_embeds = compel_proc(self.req.image_metadata.prompt)
        self.req.negative_prompt_embeds = compel_proc(self.req.image_metadata.negative_prompt)

        # generate
        images = pipeline(self.req)

        step = self.compute_pipeline_steps()
        for image in images:
            # ESRGAN
            if self.req.image_metadata.upscale_enabled:
                esrgan = ESRGANProcessor()
                esrgan.upscale_factor = self.req.image_metadata.upscale_factor
                esrgan.denoising_strength = self.req.image_metadata.upscale_denoising_strength
                esrgan.blend_strength = self.req.image_metadata.upscale_blend_strength
                upscaled_image = esrgan(image)
            else:
                upscaled_image = image

            self.update_task_progress(step)
            step = step + 1

            # GFPGAN
            if self.req.image_metadata.face_enabled:
                gfpgan = GFPGANProcessor()
                gfpgan.upscale_factor = self.req.image_metadata.upscale_factor
                gfpgan.upscaled_image = upscaled_image
                gfpgan.blend_strength = self.req.image_metadata.face_blend_strength
                image = gfpgan(image)
            else:
                image = upscaled_image

            self.update_task_progress(step)
            step = step + 1

            # Output
            collection = self.collection
            png_info = PngImagePlugin.PngInfo()
            self.req.image_metadata.save_to_png_info(png_info)

            def io_operation():
                next_image_id = utils.next_image_id(os.path.join(configuration.IMAGES_PATH, collection))
                output_path = os.path.join(collection, '{:05d}.png'.format(next_image_id))
                full_path = os.path.join(configuration.IMAGES_PATH, output_path)

                image.save(full_path, pnginfo=png_info)
                return output_path

            output_path = utils.retry_on_failure(io_operation)

            self.update_task_progress(step)
            step = step + 1

            self.image_complete.emit(output_path)

    def generate_callback(self, step: int, timestep: int, latents: torch.FloatTensor):
        if self.cancel:
            raise CancelThreadException()
        self.update_task_progress(step)

        pil_image = latents_to_pil(latents)
        pil_image = pil_image.resize((pil_image.size[0] * 8, pil_image.size[1] * 8), Image.NEAREST)
        self.image_preview.emit(pil_image)

    def update_task_progress(self, step: int):
        steps = self.compute_total_steps()
        progress_amount = (step+1) * 100 / steps
        self.task_progress.emit(progress_amount)

    def compute_pipeline_steps(self):
        steps = self.req.image_metadata.num_inference_steps
        if self.req.image_metadata.img2img_enabled:
            steps = int(self.req.image_metadata.num_inference_steps * self.req.image_metadata.img2img_strength)

        return steps

    def compute_total_steps(self):
        steps_per_image = 1
        if self.req.image_metadata.upscale_enabled:
            steps_per_image = steps_per_image + 1
        if self.req.image_metadata.face_enabled:
            steps_per_image = steps_per_image + 1

        return self.compute_pipeline_steps() + self.req.num_images_per_prompt * steps_per_image
    