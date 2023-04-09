import json
import os

import configuration
from configuration import ControlNetCondition


class ImageMetadata:
    type: str = 'txt2img'
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
    condition: str = 'Text'
    control_net_preprocess: bool = False
    control_net_model: str = ''
    control_net_scale: float = 1.0
    source_path: str = ''
    img_strength: float = 0.0
    upscale_enabled: bool = False
    upscale_factor: int = 1
    upscale_denoising_strength: float = 0.0
    upscale_blend_strength: float = 0.0
    face_enabled: bool = False
    face_blend_strength: float  = 0.0

    def load_from_settings(self, settings):
        self.type = settings.value('type')
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
        self.condition = 'Text'
        self.control_net_preprocess = False
        self.control_net_model = ''
        self.control_net_scale = 1.0
        self.source_path = ''
        self.img_strength = 0.0
        if self.type == 'img2img':
            self.condition = settings.value('condition', 'Image')
            condition = configuration.conditions.get(self.condition, None)
            if isinstance(condition, ControlNetCondition):
                self.control_net_preprocess = settings.value('control_net_preprocess', type=bool)
                self.control_net_model = settings.value('control_net_model')
                self.control_net_scale = settings.value('control_net_scale')
            self.source_path = settings.value('source_path')
            self.img_strength = float(settings.value('img_strength'))
        self.upscale_enabled = settings.value('upscale_enabled', type=bool)
        self.upscale_factor = 1
        self.upscale_denoising_strength = 0.0
        self.upscale_blend_strength = 0.0
        if self.upscale_enabled:
            self.upscale_factor = int(settings.value('upscale_factor'))
            self.upscale_denoising_strength = float(settings.value('upscale_denoising_strength'))
            self.upscale_blend_strength = float(settings.value('upscale_blend_strength'))
        self.face_enabled = settings.value('face_enabled', type=bool)
        if self.face_enabled:
            self.face_blend_strength = float(settings.value('face_blend_strength'))
    
    def load_from_image_info(self, image_info):
        if 'sd-metadata' in image_info:
            sd_metadata = json.loads(image_info['sd-metadata'])
            self.model = sd_metadata.get('model_weights', 'stable-diffusion-2-1-base')
            if 'image' in sd_metadata:
                image_data = sd_metadata['image']
                self.type = image_data.get('type', 'txt2img')
                self.safety_checker = bool(image_data.get('safety_checker', False))
                self.scheduler = image_data.get('sampler', 'k_euler_a')
                self.prompt = image_data.get('prompt', '')
                self.negative_prompt = image_data.get('negative_prompt', '')
                self.seed = int(image_data.get('seed', 5))
                self.steps = int(image_data.get('steps', 30))
                self.guidance_scale = float(image_data.get('cfg_scale', 7.5))
                self.width = int(image_data.get('width', 512))
                self.height = int(image_data.get('height', 512))
                self.condition = image_data.get('condition', 'Text')
                self.control_net_preprocess = False
                self.control_net_model = ''
                self.control_net_scale = 1.0
                self.source_path = ''
                self.img_strength = 0.0
                if self.type == 'img2img':
                    condition = configuration.conditions.get(self.condition, None)
                    if isinstance(condition, ControlNetCondition):
                        self.control_net_preprocess = image_data.get('control_net_preprocess', False)
                        self.control_net_model = image_data.get('control_net_model', '')
                        self.control_net_scale = image_data.get('control_net_scale', 1.0)
                    self.source_path = image_data.get('source_path', '')
                    self.img_strength = float(image_data.get('img_strength', 0.5))
                self.upscale_enabled = 'upscale_blend_strength' in image_data
                self.upscale_factor = 1
                self.upscale_denoising_strength = 0.0
                self.upscale_blend_strength = 0.0
                if self.upscale_enabled:
                    self.upscale_factor = int(image_data.get('upscale_factor'))
                    self.upscale_denoising_strength = float(image_data.get('upscale_denoising_strength'))
                    self.upscale_blend_strength = float(image_data.get('upscale_blend_strength'))
                self.face_enabled = 'face_blend_strength' in image_data
                if self.face_enabled:
                    self.face_blend_strength = float(image_data.get('face_blend_strength'))

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
                'type': self.type,
                'safety_checker': self.safety_checker,
                'sampler': self.scheduler,
                'condition': self.condition,
            }
        }
        if self.type == 'img2img':
            condition = configuration.conditions.get(self.condition, None)
            if isinstance(condition, ControlNetCondition):
                sd_metadata['image']['control_net_preprocess'] = self.control_net_preprocess
                sd_metadata['image']['control_net_model'] = self.control_net_model
                sd_metadata['image']['control_net_scale'] = self.control_net_scale
            sd_metadata['image']['source_path'] = self.source_path
            sd_metadata['image']['img_strength'] = self.img_strength
        if self.upscale_enabled:
            sd_metadata['image']['upscale_factor'] = self.upscale_factor
            sd_metadata['image']['upscale_denoising_strength'] = self.upscale_denoising_strength
            sd_metadata['image']['upscale_blend_strength'] = self.upscale_blend_strength
        if self.face_enabled:
            sd_metadata['image']['face_blend_strength'] = self.face_blend_strength

        png_info.add_text('Dream',
            '"{:s} [{:s}]" -s {:d} -S {:d} -W {:d} -H {:d} -C {:f} -A {:s}'.format(
                self.prompt, self.negative_prompt, self.num_inference_steps, self.seed, self.width, self.height, self.guidance_scale, self.scheduler
            ))
        png_info.add_text('sd-metadata', json.dumps(sd_metadata))
