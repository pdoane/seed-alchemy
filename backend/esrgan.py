# Based on https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/inference_realesrgan.py

import os
import warnings

import numpy as np
from PIL import Image

from . import config, utils
from .device import default_device


class ESRGANProcessor:
    def __init__(self):
        self.model_name = None
        self.denoising_strength = None
        self.tile_size = None
        self.tile_pad = None
        self.pre_pad = None
        self.float32 = None
        self.esrgan = None

    def __call__(
        self,
        image: Image.Image,
        model_name: str = "realesr-general-x4v3",
        upscale_factor: int = 2,
        denoising_strength: float = 0.5,
        blend_strength: float = 1.0,
        tile_size: int = 512,
        tile_pad: int = 10,
        pre_pad: int = 0,
        float32: bool = True,
    ) -> Image.Image:
        # Load
        if (
            self.model_name != model_name
            or self.denoising_strength != denoising_strength
            or self.tile_size != tile_size
            or self.tile_pad != tile_pad
            or self.pre_pad != pre_pad
            or self.float32 != float32
        ):
            self.esrgan = None

        if not self.esrgan:
            self.esrgan = self.load(model_name, denoising_strength, tile_size, tile_pad, pre_pad, float32)

            self.model_name = model_name
            self.denoising_strength = denoising_strength
            self.tile_size = tile_size
            self.tile_pad = tile_pad
            self.pre_pad = pre_pad
            self.float32 = float32

        # Process
        bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

        output, _ = self.esrgan.enhance(
            bgr_image_array,
            outscale=upscale_factor,
        )

        image2 = Image.fromarray(output[..., ::-1])

        if blend_strength < 1.0:
            if image2.size != image.size:
                image = image.resize(image2.size, Image.Resampling.LANCZOS)
            image = Image.blend(image, image2, blend_strength)
        else:
            image = image2

        return image

    @classmethod
    def load(
        cls, model_name: str, denoising_strength: float, tile_size: int, tile_pad: int, pre_pad: int, float32: bool
    ):
        print("Loading ESRGAN", model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        if model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]
        elif model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"]
        elif model_name == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"]
        elif model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]
        elif model_name == "realesr-animevideov3":  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
            netscale = 4
            urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"]
        elif model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
            netscale = 4
            urls = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            ]

        model_paths = []
        for url in urls:
            model_path = config.get_cache_path("realesrgan", os.path.basename(url))
            model_paths.append(model_path)
            if not os.path.exists(model_path):
                utils.download_file(url, model_path)

        dni_weight = None
        if model_name == "realesr-general-x4v3" and denoising_strength != 1:
            dni_weight = [denoising_strength, 1 - denoising_strength]
            model_path = model_paths
        else:
            model_path = model_paths[0]

        return RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=tile_size,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=float32,
            device=default_device(),
        )
