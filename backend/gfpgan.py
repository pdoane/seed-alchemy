# Based on https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/inference_realesrgan.py

import os
import warnings

import numpy as np
from PIL import Image

from . import config, utils
from .device import default_device


class GFPGANProcessor:
    def __init__(self):
        self.model_name = None
        self.upscale_factor = None
        self.gfpgan = None

    def __call__(
        self,
        image: Image.Image,
        model_name: str = "GFPGANv1.4",
        upscale_factor: int = 2,
        upscaled_image: Image.Image = None,
        blend_strength: float = 1.0,
    ) -> Image.Image:
        # Load
        if self.model_name != model_name or self.upscale_factor != upscale_factor:
            self.gfpgan = None

        if not self.gfpgan:
            self.gfpgan = self.load(model_name, upscale_factor)

            self.model_name = model_name
            self.upscale_factor = upscale_factor

        # Enhance faces on unscaled image
        bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

        _, _, output = self.gfpgan.enhance(
            bgr_image_array,
            has_aligned=False,
            only_center_face=False,
            paste_back=False,
        )

        # Paste faces onto upscaled image
        image = upscaled_image
        bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

        self.gfpgan.face_helper.get_inverse_affine(None)
        output = self.gfpgan.face_helper.paste_faces_to_input_image(upsample_img=bgr_image_array)

        image2 = Image.fromarray(output[..., ::-1])

        # Blend with upscaled image
        if blend_strength < 1.0:
            if image2.size != image.size:
                image = image.resize(image2.size, Image.Resampling.LANCZOS)
            image = Image.blend(image, image2, blend_strength)
        else:
            image = image2

        return image

    @classmethod
    def load(cls, model_name: str, upscale_factor: int):
        print("Loading GFPGAN", model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from .gfpgan_util import GFPGANer

            if model_name == "GFPGANv1.2":
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth"
                arch = "clean"
            elif model_name == "GFPGANv1.3":
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
                arch = "clean"
            elif model_name == "GFPGANv1.4":
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
                arch = "clean"
            elif model_name == "RestoreFormer":
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
                arch = "RestoreFormer"

            model_path = config.get_cache_path("gfpgan", os.path.basename(url))
            if not os.path.exists(model_path):
                utils.download_file(url, model_path)

            return GFPGANer(
                model_path=model_path,
                upscale=upscale_factor,
                arch=arch,
                channel_multiplier=2,
                bg_upsampler=None,
                model_rootpath=config.get_cache_path("gfpgan", None),
                device=default_device(),
            )
