import os
import warnings
from abc import ABC, abstractmethod

import configuration
import cv2
import numpy as np
import torch
import utils
from controlnet_aux import (CannyDetector, HEDdetector, MidasDetector,
                            MLSDdetector, OpenposeDetector)
from controlnet_aux import util as controlnet_utils
from controlnet_aux.midas.api import MiDaSInference
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation


# Fix issues in controlnet_aux 0.0.2 requiring Cuda
class MidasDetector:
    def __init__(self, model_type="dpt_hybrid", model_path=None):
        self.model = MiDaSInference(model_type=model_type, model_path=model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, model_type="dpt_hybrid", filename=None, cache_dir=None):
        filename = filename or "annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
        model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
        return cls(model_type=model_type, model_path=model_path)
        
    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        
        input_type = "np"
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
            input_type = "pil"
            
        input_image = controlnet_utils.HWC3(input_image)
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float()
            if torch.cuda.is_available():
                image_depth = image_depth.cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        
        if input_type == "pil":
            depth_image = Image.fromarray(depth_image)
            depth_image = depth_image.convert("RGB")
            normal_image = Image.fromarray(normal_image)
        
        return depth_image, normal_image

class ProcessorBase(ABC):
    @abstractmethod
    def __call__(self, image: Image.Image) -> Image.Image:
        pass

class CannyProcessor(ProcessorBase):
    low_threshold = 100
    high_threshold = 100

    def __init__(self) -> None:
        super().__init__()
        self.canny = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.canny is None:
            self.canny = CannyDetector()
        image = self.canny(image, self.low_threshold, self.high_threshold)
        return image

class DepthProcessor(ProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.midas = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.midas is None:
            self.midas = MidasDetector.from_pretrained('lllyasviel/ControlNet')
        image, _ = self.midas(image)
        return image

class NormalProcessor(ProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.midas = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.midas is None:
            self.midas = MidasDetector.from_pretrained('lllyasviel/ControlNet')
        _, image = self.midas(image)
        return image

class HedProcessor(ProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.hed = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.hed is None:
            self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        image = self.hed(image)
        return image

class MlsdProcessor(ProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.mlsd = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.mlsd is None:
            self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        image = self.mlsd(image)
        return image

class OpenposeProcessor(ProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.openpose = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.openpose is None:
            self.openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        image = self.openpose(image)
        return image

class ScribbleProcessor(ProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.hed = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.hed is None:
            self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        image = self.hed(image, scribble=True)
        return image

class SegProcessor(ProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.image_processor = None
        self.image_segmentor = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.image_processor is None:
            self.image_processor = AutoImageProcessor.from_pretrained('openmmlab/upernet-convnext-small')
        if self.image_segmentor is None:
            self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained('openmmlab/upernet-convnext-small')
        pixel_values = self.image_processor(image, return_tensors='pt').pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(controlnet_utils.ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)
        return image

class ESRGANProcessor(ProcessorBase):
    # Based on https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/inference_realesrgan.py

    model_name: str = 'realesr-general-x4v3'
    upscale_factor: int = 2
    denoising_strength: float = 0.5
    blend_strength: float = 1.0
    tile_size: int = 512
    tile_pad: int = 10
    pre_pad: int = 0
    float32: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.esrgan = None

    def load(self):
        with warnings.catch_warnings(), utils.ChangeDirectory(configuration.MODELS_PATH):
            warnings.simplefilter('ignore')
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact

            if self.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            elif self.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
            elif self.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
                urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
            elif self.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
                urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
            elif self.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
                netscale = 4
                urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
            elif self.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
                netscale = 4
                urls = [
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
                ]

            model_paths = []
            for url in urls:
                model_path = os.path.join('realesrgan', os.path.basename(url))
                model_paths.append(model_path)
                if not os.path.exists(model_path):
                    utils.download_file(url, model_path)

            dni_weight = None
            if self.model_name == 'realesr-general-x4v3' and self.denoising_strength != 1:
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
                half=not self.float32)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.esrgan is None:
            self.load()

        bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

        output, _ = self.esrgan.enhance(
            bgr_image_array,
            outscale = self.upscale_factor,
        )

        image2 = Image.fromarray(output[..., ::-1])

        if self.blend_strength < 1.0:
            if image2.size != image.size:
                image = image.resize(image2.size)
            image = Image.blend(image, image2, self.blend_strength)
        else:
            image = image2

        return image

class GFPGANProcessor(ProcessorBase):
    # Based on https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/inference_realesrgan.py

    model_name: str = 'GFPGANv1.4'
    upscale_factor: int = 2
    upscaled_image: Image.Image = None
    blend_strength: float = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.esrgan = None
        self.gfpgan = None

    def load(self):
        with warnings.catch_warnings(), utils.ChangeDirectory(configuration.MODELS_PATH):
            warnings.simplefilter('ignore')
            from gfpgan import GFPGANer

            if self.model_name == 'GFPGANv1.2':
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth'
                arch = 'clean'
            elif self.model_name == 'GFPGANv1.3':
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
                arch = 'clean'
            elif self.model_name == 'GFPGANv1.4':
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
                arch = 'clean'
            elif self.model_name == 'RestoreFormer':
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
                arch = 'RestoreFormer'

            model_path = os.path.join('gfpgan', os.path.basename(url))
            if not os.path.exists(model_path):
                utils.download_file(url, model_path)

            self.gfpgan = GFPGANer(
                model_path=model_path,
                upscale=self.upscale_factor,
                arch=arch,
                channel_multiplier=2,
                bg_upsampler=self.esrgan)

    def __call__(self, image: Image.Image) -> Image.Image:
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
                image = image.resize(image2.size)
            image = Image.blend(image, image2, self.blend_strength)
        else:
            image = image2

        return image
