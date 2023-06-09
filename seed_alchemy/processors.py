import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LeresDetector,
    LineartAnimeDetector,
    LineartDetector,
    MediapipeFaceDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
    ZoeDetector,
)
from controlnet_aux.util import HWC3, ade_palette
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from . import configuration, utils


class ProcessorBase(ABC):
    params = []

    @abstractmethod
    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        pass


@dataclass
class ProcessorParameter:
    type: type
    name: str
    min: float
    max: float
    value: float
    step: float = 1


class CannyProcessor(ProcessorBase):
    params = [
        ProcessorParameter(type=int, name="Low", min=1, max=255, value=100),
        ProcessorParameter(type=int, name="High", min=1, max=255, value=200),
    ]

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.canny = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.canny is None:
            self.canny = CannyDetector()
        image = self.canny(image, params[0], params[1])
        return image


class DepthLeresProcessor(ProcessorBase):
    params = [
        ProcessorParameter(type=float, name="Near", min=0.0, max=1.0, value=0.0, step=0.01),
        ProcessorParameter(type=float, name="Background", min=0.0, max=1.0, value=0.0, step=0.01),
    ]

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.leres = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.leres is None:
            self.leres = LeresDetector.from_pretrained("lllyasviel/Annotators")
            self.leres.to(self.device)
        image = self.leres(image, thr_a=params[0] * 100, thr_b=params[1] * 100)
        return image


class DepthLeresBoostProcessor(ProcessorBase):
    params = [
        ProcessorParameter(type=float, name="Near", min=0.0, max=1.0, value=0.0, step=0.01),
        ProcessorParameter(type=float, name="Background", min=0.0, max=1.0, value=0.0, step=0.01),
    ]

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.leres = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.leres is None:
            self.leres = LeresDetector.from_pretrained("lllyasviel/Annotators")
            self.leres.to(self.device)
        image = self.leres(image, thr_a=params[0] * 100, thr_b=params[1] * 100, boost=True)
        return image


class DepthMidasProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.midas = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.midas is None:
            self.midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            self.midas.to(self.device)
        image = self.midas(image)
        return image


class DepthZoeProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.zoe = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.zoe is None:
            self.zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
            self.zoe.to(self.device)
        image = self.zoe(image)
        return image


class LineartProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lineart = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.lineart is None:
            self.lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
            self.lineart.to(self.device)
        image = self.lineart(image)
        return image


class LineartCoarseProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lineart = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.lineart is None:
            self.lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
            self.lineart.to(self.device)
        image = self.lineart(image, coarse=True)
        return image


class LineartAnimeProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lineart_anime = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.lineart_anime is None:
            self.lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
            self.lineart_anime.to(self.device)
        image = self.lineart_anime(image)
        return image


class MediapipeFaceProcessor(ProcessorBase):
    params = [
        ProcessorParameter(type=int, name="Max Faces", min=1, max=10, value=1),
        ProcessorParameter(type=float, name="Confidence", min=0.01, max=1.0, value=0.5, step=0.01),
    ]

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.face_detector = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.face_detector is None:
            self.face_detector = MediapipeFaceDetector()
        image = self.face_detector(image, max_faces=params[0], min_confidence=params[1])
        return image


class MlsdProcessor(ProcessorBase):
    params = [
        ProcessorParameter(type=float, name="Value", min=0.01, max=2.0, value=0.1, step=0.01),
        ProcessorParameter(type=float, name="Distance", min=0.01, max=20.0, value=0.1, step=0.01),
    ]

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mlsd = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.mlsd is None:
            self.mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
            self.mlsd.to(self.device)
        image = self.mlsd(image, thr_v=params[0], thr_d=params[1])
        return image


class NormalBaeProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.normal_bae = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.normal_bae is None:
            self.normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
            self.normal_bae.to(self.device)
        image = self.normal_bae(image)
        return image


class NormalMidasProcessor(ProcessorBase):
    params = [ProcessorParameter(type=float, name="Background", min=0.0, max=1.0, value=0.4, step=0.01)]

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.midas = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.midas is None:
            self.midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            self.midas.to(self.device)
        _, image = self.midas(image, bg_th=params[0], depth_and_normal=True)
        return image


class OpenposeProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.openpose = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.openpose is None:
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            self.openpose.to(self.device)
        image = self.openpose(image, include_body=True, include_hand=False, include_face=False)
        return image


class OpenposeFaceProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.openpose = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.openpose is None:
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            self.openpose.to(self.device)
        image = self.openpose(image, include_body=True, include_hand=False, include_face=True)
        return image


class OpenposeFaceOnlyProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.openpose = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.openpose is None:
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            self.openpose.to(self.device)
        image = self.openpose(image, include_body=False, include_hand=False, include_face=True)
        return image


class OpenposeFullProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.openpose = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.openpose is None:
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            self.openpose.to(self.device)
        image = self.openpose(image, include_body=True, include_hand=True, include_face=True)
        return image


class OpenposeHandProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.openpose = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.openpose is None:
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            self.openpose.to(self.device)
        image = self.openpose(image, include_body=True, include_hand=True, include_face=False)
        return image


class ScribbleHEDProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.hed = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.hed is None:
            self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
            self.hed.to(self.device)
        image = self.hed(image, scribble=True)
        return image


class ScribblePIDIProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pidi = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.pidi is None:
            self.pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            self.pidi.to(self.device)
        image = self.pidi(image, scribble=True)
        return image


class ScribbleXDoGProcessor(ProcessorBase):
    params = [ProcessorParameter(type=int, name="Threshold", min=1, max=64, value=32)]

    def __init__(self, device):
        super().__init__()
        self.device = device

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        thr = params[0]

        img = np.array(image)
        img = HWC3(img)

        g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
        g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
        dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
        result = np.zeros_like(img, dtype=np.uint8)
        result[2 * (255 - dog) > thr] = 255

        img = Image.fromarray(result)
        img = img.convert("RGB")
        return img


class ShuffleProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.content = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.content is None:
            self.content = ContentShuffleDetector()
        image = self.content(image)
        return image


class SegProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.image_processor = None
        self.image_segmentor = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.image_processor is None:
            self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        if self.image_segmentor is None:
            self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)
        return image


class SoftEdgeHEDProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.hed = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.hed is None:
            self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
            self.hed.to(self.device)
        image = self.hed(image)
        return image


class SoftEdgeHEDSafeProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.hed = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.hed is None:
            self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
            self.hed.to(self.device)
        image = self.hed(image, safe=True)
        return image


class SoftEdgePIDIProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pidi = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.pidi is None:
            self.pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            self.pidi.to(self.device)
        image = self.pidi(image)
        return image


class SoftEdgePIDISafeProcessor(ProcessorBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pidi = None

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.pidi is None:
            self.pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            self.pidi.to(self.device)
        image = self.pidi(image, safe=True)
        return image


class ESRGANProcessor(ProcessorBase):
    # Based on https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/inference_realesrgan.py

    model_name: str = "realesr-general-x4v3"
    upscale_factor: int = 2
    denoising_strength: float = 0.5
    blend_strength: float = 1.0
    tile_size: int = 512
    tile_pad: int = 10
    pre_pad: int = 0
    float32: bool = True

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.esrgan = None

    def load(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact

            if self.model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]
            elif self.model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"]
            elif self.model_name == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet model with 6 blocks
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
                urls = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
                ]
            elif self.model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
                urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]
            elif self.model_name == "realesr-animevideov3":  # x4 VGG-style model (XS size)
                model = SRVGGNetCompact(
                    num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"
                )
                netscale = 4
                urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"]
            elif self.model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
                model = SRVGGNetCompact(
                    num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
                )
                netscale = 4
                urls = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                ]

            model_paths = []
            for url in urls:
                model_path = os.path.join(configuration.MODELS_PATH, "realesrgan", os.path.basename(url))
                model_paths.append(model_path)
                if not os.path.exists(model_path):
                    utils.download_file(url, model_path)

            dni_weight = None
            if self.model_name == "realesr-general-x4v3" and self.denoising_strength != 1:
                dni_weight = [self.denoising_strength, 1 - self.denoising_strength]
                model_path = model_paths
            else:
                model_path = model_paths[0]

            self.esrgan = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                dni_weight=dni_weight,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=not self.float32,
                device=self.device,
            )

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.esrgan is None:
            self.load()

        bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

        output, _ = self.esrgan.enhance(
            bgr_image_array,
            outscale=self.upscale_factor,
        )

        image2 = Image.fromarray(output[..., ::-1])

        if self.blend_strength < 1.0:
            if image2.size != image.size:
                image = image.resize(image2.size, Image.Resampling.LANCZOS)
            image = Image.blend(image, image2, self.blend_strength)
        else:
            image = image2

        return image


class GFPGANProcessor(ProcessorBase):
    # Based on https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/inference_realesrgan.py

    model_name: str = "GFPGANv1.4"
    upscale_factor: int = 2
    upscaled_image: Image.Image = None
    blend_strength: float = 1.0

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.esrgan = None
        self.gfpgan = None

    def load(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from .gfpgan_util import GFPGANer

            if self.model_name == "GFPGANv1.2":
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth"
                arch = "clean"
            elif self.model_name == "GFPGANv1.3":
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
                arch = "clean"
            elif self.model_name == "GFPGANv1.4":
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
                arch = "clean"
            elif self.model_name == "RestoreFormer":
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
                arch = "RestoreFormer"

            model_path = os.path.join(configuration.MODELS_PATH, "gfpgan", os.path.basename(url))
            if not os.path.exists(model_path):
                utils.download_file(url, model_path)

            self.gfpgan = GFPGANer(
                model_path=model_path,
                upscale=self.upscale_factor,
                arch=arch,
                channel_multiplier=2,
                bg_upsampler=self.esrgan,
                model_rootpath=os.path.join(configuration.MODELS_PATH, "gfpgan"),
                device=self.device,
            )

    def __call__(self, image: Image.Image, params: list[float]) -> Image.Image:
        if self.gfpgan is None:
            self.load()

        # Enhance faces on unscaled image
        bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

        _, _, output = self.gfpgan.enhance(
            bgr_image_array,
            has_aligned=False,
            only_center_face=False,
            paste_back=False,
        )

        # Paste faces onto upscaled image
        image = self.upscaled_image
        bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

        self.gfpgan.face_helper.get_inverse_affine(None)
        output = self.gfpgan.face_helper.paste_faces_to_input_image(upsample_img=bgr_image_array)

        image2 = Image.fromarray(output[..., ::-1])

        # Blend with upscaled image
        if self.blend_strength < 1.0:
            if image2.size != image.size:
                image = image.resize(image2.size, Image.Resampling.LANCZOS)
            image = Image.blend(image, image2, self.blend_strength)
        else:
            image = image2

        return image
