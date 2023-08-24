from diffusers import AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor

from .device import default_device, default_dtype
from .types import BaseModelType


class TinyVAE:
    def __init__(self):
        self.base_model_type = None
        self.vae = None
        self.image_processsor = None

    def load(self, base_model_type):
        if self.base_model_type != base_model_type:
            self.unload()

        if not self.vae:
            device = default_device()
            torch_dtype = default_dtype()

            if base_model_type in [BaseModelType.SDXL, BaseModelType.SDXL_REFINER]:
                repo_id = "madebyollin/taesdxl"
            else:
                repo_id = "madebyollin/taesd"
            vae = AutoencoderTiny.from_pretrained(repo_id, torch_dtype=torch_dtype)
            vae.to(device)
            image_processsor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)

            self.base_model_type = base_model_type
            self.vae = vae
            self.image_processsor = image_processsor

    def unload(self):
        self.vae = None
        self.image_processsor = None

    def decode(self, latents):
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        return self.image_processsor.postprocess(image, output_type="pil", do_denormalize=[True])[0]
