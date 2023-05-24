import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json
from PIL import Image

from . import configuration


@dataclass_json
@dataclass
class ControlNetMetadata:
    model: str = ''
    image_source: str = ''
    preprocessor: Optional[str] = None
    params: Optional[List[float]] = field(default_factory=list)
    scale: float = 1.0
    
class ImageMetadata:
    model: str = 'stabilityai/stable-diffusion-2-1-base'
    safety_checker: bool = True
    scheduler: str = 'k_euler_a'
    path: str = ''
    prompt: str = ''
    negative_prompt: str = ''
    seed: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    img2img_enabled: bool = False
    img2img_source: str = ''
    img2img_noise: float = 1.0
    control_net_enabled: bool = False
    control_net_guidance_start: float = 0.0
    control_net_guidance_end: float = 1.0
    control_nets: list[ControlNetMetadata] = []
    upscale_enabled: bool = False
    upscale_factor: int = 1
    upscale_denoising_strength: float = 0.0
    upscale_blend_strength: float = 0.0
    face_enabled: bool = False
    face_blend_strength: float = 0.0
    high_res_enabled: bool = False
    high_res_factor: float = 1.0
    high_res_steps: int = 0
    high_res_guidance_scale: float = 7.5
    high_res_noise: float = 0.0

    def load_from_settings(self, settings):
        self.model = settings.value('model')
        self.safety_checker = settings.value('safety_checker', type=bool)
        self.scheduler = settings.value('scheduler')
        self.prompt = settings.value('prompt')
        self.negative_prompt = settings.value('negative_prompt')

        self.seed = int(settings.value('seed'))
        self.num_inference_steps = int(settings.value('num_inference_steps'))
        self.guidance_scale = float(settings.value('guidance_scale'))
        self.width = int(settings.value('width'))
        self.height = int(settings.value('height'))

        self.img2img_enabled = settings.value('img2img_enabled', type=bool)
        if self.img2img_enabled:
            self.img2img_source = settings.value('img2img_source')
            self.img2img_noise = float(settings.value('img2img_noise'))

        self.control_net_enabled = settings.value('control_net_enabled', type=bool)
        self.control_nets = []
        if self.control_net_enabled:
            self.control_net_guidance_start = settings.value('control_net_guidance_start')
            self.control_net_guidance_end = settings.value('control_net_guidance_end')
            self.control_nets = [ControlNetMetadata.from_dict(item) for item in json.loads(settings.value('control_nets'))]

        self.upscale_enabled = settings.value('upscale_enabled', type=bool)
        if self.upscale_enabled:
            self.upscale_factor = int(settings.value('upscale_factor'))
            self.upscale_denoising_strength = float(settings.value('upscale_denoising_strength'))
            self.upscale_blend_strength = float(settings.value('upscale_blend_strength'))

        self.face_enabled = settings.value('face_enabled', type=bool)
        if self.face_enabled:
            self.face_blend_strength = float(settings.value('face_blend_strength'))
        
        self.high_res_enabled = settings.value('high_res_enabled', type=bool)
        if self.high_res_enabled:
            self.high_res_factor = settings.value('high_res_factor', type=float)
            self.high_res_steps = settings.value('high_res_steps', type=int)
            self.high_res_guidance_scale = settings.value('high_res_guidance_scale', type=float)
            self.high_res_noise = settings.value('high_res_noise', type=float)


    def load_from_image(self, image: Image.Image):
        self.width = image.width
        self.height = image.height
        
        image_info = image.info
        if 'sd-metadata' in image_info:
            sd_metadata = json.loads(image_info['sd-metadata'])
            self.model = sd_metadata.get('model_weights', 'stable-diffusion-2-1-base')
            if 'image' in sd_metadata:
                image_data = sd_metadata['image']
                self.safety_checker = bool(image_data.get('safety_checker', False))
                self.scheduler = image_data.get('sampler', 'k_euler_a')
                self.prompt = image_data.get('prompt', '')
                if isinstance(self.prompt, list):
                    self.prompt = self.prompt[0]
                if 'prompt' in self.prompt:
                    self.prompt = self.prompt['prompt']
                self.negative_prompt = image_data.get('negative_prompt', '')

                self.seed = int(image_data.get('seed', 5))
                self.steps = int(image_data.get('steps', 30))
                self.guidance_scale = float(image_data.get('cfg_scale', 7.5))
                self.width = int(image_data.get('width', 512))
                self.height = int(image_data.get('height', 512))

                self.img2img_enabled = 'img2img_source' in image_data
                if self.img2img_enabled:
                    self.img2img_source = str(image_data.get('img2img_source', ''))
                    self.img2img_noise = float(image_data.get('img2img_noise', 0.5))

                self.control_net_enabled = 'control_nets' in image_data
                if self.control_net_enabled:
                    self.control_net_guidance_start = image_data.get('control_net_guidance_start', 0.0)
                    self.control_net_guidance_end = image_data.get('control_net_guidance_end', 1.0)
                    self.control_nets = [ControlNetMetadata.from_dict(item) for item in image_data.get('control_nets', '[]')]

                self.upscale_enabled = 'upscale_blend_strength' in image_data
                if self.upscale_enabled:
                    self.upscale_factor = int(image_data.get('upscale_factor', 1))
                    self.upscale_denoising_strength = float(image_data.get('upscale_denoising_strength', 0.0))
                    self.upscale_blend_strength = float(image_data.get('upscale_blend_strength', 0.0))

                self.face_enabled = 'face_blend_strength' in image_data
                if self.face_enabled:
                    self.face_blend_strength = float(image_data.get('face_blend_strength', 0.0))
                
                self.high_res_enabled = 'high_res_factor' in image_data
                if self.high_res_enabled:
                    self.high_res_factor = float(image_data.get('high_res_factor', 1.0))
                    self.high_res_steps = int(image_data.get('high_res_steps', 0))
                    self.high_res_guidance_scale = float(image_data.get('high_res_guidance_scale', 7.5))
                    self.high_res_noise = float(image_data.get('high_res_noise', 0.0))

    def save_to_png_info(self, png_info):
        sd_metadata = {
            'model': 'stable diffusion',
            'model_weights': os.path.basename(self.model),
            'model_hash': '',    # TODO
            'app_id': configuration.APP_NAME,
            'APP_VERSION': configuration.APP_VERSION,
            'image': {
                'prompt': self.prompt,
                'negative_prompt': self.negative_prompt,
                'steps': str(self.num_inference_steps),
                'cfg_scale': str(self.guidance_scale),
                'height': str(self.height),
                'width': str(self.width),
                'seed': str(self.seed),
                'safety_checker': self.safety_checker,
                'sampler': self.scheduler,
            }
        }
        if self.img2img_enabled:
            sd_metadata['image']['img2img_source'] = self.img2img_source
            sd_metadata['image']['img2img_noise'] = self.img2img_noise
        if self.control_net_enabled:
            sd_metadata['image']['control_net_guidance_start'] = self.control_net_guidance_start
            sd_metadata['image']['control_net_guidance_end'] = self.control_net_guidance_end
            sd_metadata['image']['control_nets'] = [control_net.to_dict() for control_net in self.control_nets]
        if self.upscale_enabled:
            sd_metadata['image']['upscale_factor'] = self.upscale_factor
            sd_metadata['image']['upscale_denoising_strength'] = self.upscale_denoising_strength
            sd_metadata['image']['upscale_blend_strength'] = self.upscale_blend_strength
        if self.face_enabled:
            sd_metadata['image']['face_blend_strength'] = self.face_blend_strength
        if self.high_res_enabled:
            sd_metadata['image']['high_res_factor'] = self.high_res_factor
            sd_metadata['image']['high_res_steps'] = self.high_res_steps
            sd_metadata['image']['high_res_guidance_scale'] = self.high_res_guidance_scale
            sd_metadata['image']['high_res_noise'] = self.high_res_noise

        png_info.add_text('Dream',
            '"{:s} [{:s}]" -s {:d} -S {:d} -W {:d} -H {:d} -C {:f} -A {:s}'.format(
                self.prompt, self.negative_prompt, self.num_inference_steps, self.seed, self.width, self.height, self.guidance_scale, self.scheduler
            ))
        png_info.add_text('sd-metadata', json.dumps(sd_metadata))
