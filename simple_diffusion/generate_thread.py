import gc
import os
import traceback

import configuration
import torch
import utils
from compel import Compel, PromptParser
from image_metadata import ImageMetadata
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

def align_down(n: int, align: int) -> int:
    return align * (n // align)

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
        self.step = 0
        self.collection = settings.value('collection')
        self.reduce_memory = settings.value('reduce_memory', type=bool)
        self.req = GenerateRequest()
        self.req.image_metadata = ImageMetadata()
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
            source_path = self.req.image_metadata.img2img_source
            full_path = os.path.join(configuration.IMAGES_PATH, source_path)
            with Image.open(full_path) as image:
                image = image.convert('RGB')
                # Delay resize until generation for enhance operations
                self.req.source_image = image.copy()

        # Conditioning images
        if self.req.image_metadata.control_net_enabled:
            for control_net_meta in self.req.image_metadata.control_nets:
                source_path = control_net_meta.image_source
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

        # prompt parsing
        pp = PromptParser()
        conjunction = pp.parse_conjunction(self.req.image_metadata.prompt)

        # loras
        loras = []
        for lora_weight in conjunction.lora_weights:
            loras.append((lora_weight.model, lora_weight.weight))
        pipeline.set_loras(loras)

        # prompt weighting
        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, textual_inversion_manager=pipeline.textual_inversion_manager)
        self.req.prompt_embeds = compel_proc(self.req.image_metadata.prompt)
        self.req.negative_prompt_embeds = compel_proc(self.req.image_metadata.negative_prompt)

        # generate
        if self.req.image_metadata.img2img_enabled and self.req.image_metadata.img2img_strength == 0.0:
            images = [self.req.source_image]
        else:
            if self.req.source_image is not None:
                self.req.source_image = self.req.source_image.resize((self.req.image_metadata.width, self.req.image_metadata.height))
            images = pipeline(self.req)

        for image in images:
            loop_count = 2 if self.req.image_metadata.high_res_enabled else 1

            for i in range(loop_count):
                # ESRGAN
                if self.req.image_metadata.upscale_enabled:
                    esrgan = ESRGANProcessor()
                    esrgan.upscale_factor = self.req.image_metadata.upscale_factor
                    esrgan.denoising_strength = self.req.image_metadata.upscale_denoising_strength
                    esrgan.blend_strength = self.req.image_metadata.upscale_blend_strength
                    upscaled_image = esrgan(image)
                    self.next_step()
                else:
                    upscaled_image = image

                # GFPGAN
                if self.req.image_metadata.face_enabled:
                    gfpgan = GFPGANProcessor()
                    gfpgan.upscale_factor = self.req.image_metadata.upscale_factor
                    gfpgan.upscaled_image = upscaled_image
                    gfpgan.blend_strength = self.req.image_metadata.face_blend_strength
                    image = gfpgan(image)
                    self.next_step()
                else:
                    image = upscaled_image
                
                # High Resolution
                if i == 0 and self.req.image_metadata.high_res_enabled:
                    high_res_width = align_down(int(self.req.image_metadata.width * self.req.image_metadata.high_res_factor), 8)
                    high_res_height = align_down(int(self.req.image_metadata.height * self.req.image_metadata.high_res_factor), 8)
                    source_image = image.resize((high_res_width, high_res_height))

                    high_res_req = GenerateRequest()
                    high_res_req.image_metadata = ImageMetadata()
                    high_res_req.source_image = source_image
                    high_res_req.image_metadata.num_inference_steps = self.req.image_metadata.high_res_steps
                    high_res_req.image_metadata.guidance_scale = self.req.image_metadata.high_res_guidance_scale
                    high_res_req.image_metadata.width = high_res_width
                    high_res_req.image_metadata.height = high_res_height
                    high_res_req.image_metadata.img2img_enabled = True
                    high_res_req.image_metadata.img2img_strength = self.req.image_metadata.high_res_noise
                    high_res_req.num_images_per_prompt = 1
                    high_res_req.generator = torch.Generator().manual_seed(self.req.image_metadata.seed)
                    high_res_req.prompt_embeds = self.req.prompt_embeds
                    high_res_req.negative_prompt_embeds = self.req.negative_prompt_embeds
                    high_res_req.callback = self.req.callback

                    image = pipeline(high_res_req)[0]

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

            self.next_step()

            self.image_complete.emit(output_path)

    def generate_callback(self, step: int, timestep: int, latents: torch.FloatTensor):
        self.next_step()

        pil_image = latents_to_pil(latents)
        pil_image = pil_image.resize((pil_image.size[0] * 8, pil_image.size[1] * 8), Image.NEAREST)
        self.image_preview.emit(pil_image)

    def next_step(self):
        if self.cancel:
            raise CancelThreadException()

        self.step += 1
        steps = self.compute_total_steps()
        progress_amount = self.step * 100 / steps
        self.task_progress.emit(progress_amount)

    def compute_total_steps(self):
        loop_count = 2 if self.req.image_metadata.high_res_enabled else 1

        steps_per_image = 1
        if self.req.image_metadata.upscale_enabled:
            steps_per_image += loop_count
        if self.req.image_metadata.face_enabled:
            steps_per_image += loop_count
        if self.req.image_metadata.high_res_enabled:
            steps_per_image += int(self.req.image_metadata.high_res_steps * self.req.image_metadata.high_res_noise)

        pipeline_steps = self.req.image_metadata.num_inference_steps
        if self.req.image_metadata.img2img_enabled:
            pipeline_steps = int(pipeline_steps * self.req.image_metadata.img2img_strength)

        return pipeline_steps + self.req.num_images_per_prompt * steps_per_image
    