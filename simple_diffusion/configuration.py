from dataclasses import dataclass

from diffusers import (DDIMScheduler, DDPMScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, SchedulerMixin, UniPCMultistepScheduler)
from processors import (CannyProcessor, DepthProcessor, HedProcessor,
                        MlsdProcessor, NormalProcessor, OpenposeProcessor,
                        ProcessorBase, ScribbleProcessor, SegProcessor)
from PySide6.QtCore import QSize

APP_NAME = 'SimpleDiffusion'
APP_VERSION = 0.1

IMAGES_PATH = 'images'
THUMBNAILS_PATH = 'thumbnails'
MODELS_PATH = '.models'

ICON_SIZE = QSize(24, 24)

@dataclass
class ControlNetModel:
    repo_id: str
    preprocessors: list[ProcessorBase]

control_net_models: dict[str, ControlNetModel] = {
    'Canny': ControlNetModel('lllyasviel/sd-controlnet-canny', [CannyProcessor]),
    'Depth': ControlNetModel('lllyasviel/sd-controlnet-depth', [DepthProcessor]),
    'Normal': ControlNetModel('lllyasviel/sd-controlnet-normal', [NormalProcessor]),
    'HED': ControlNetModel('lllyasviel/sd-controlnet-hed', [HedProcessor]),
    'M-LSD': ControlNetModel('lllyasviel/sd-controlnet-mlsd', [MlsdProcessor]),
    'Openpose': ControlNetModel('lllyasviel/sd-controlnet-openpose', [OpenposeProcessor]),
    'Scribble': ControlNetModel('lllyasviel/sd-controlnet-scribble', [ScribbleProcessor]),
    'Segmentation': ControlNetModel('lllyasviel/sd-controlnet-seg', [SegProcessor]),
}

schedulers: dict[str, SchedulerMixin] = {
    'ddim': DDIMScheduler,
    'ddpm': DDPMScheduler,
    'deis_multi': DEISMultistepScheduler,
    # 'dpm_multi': DPMSolverMultistepScheduler,
    # 'dpm': DPMSolverSinglestepScheduler,
    # 'k_dpm_2': KDPM2DiscreteScheduler,
    # 'k_dpm_2_a': KDPM2AncestralDiscreteScheduler,
    'k_euler': EulerDiscreteScheduler,
    'k_euler_a': EulerAncestralDiscreteScheduler,
    'k_heun': HeunDiscreteScheduler,
    'k_lms': LMSDiscreteScheduler,
    'pndm': PNDMScheduler,
    'uni_pc': UniPCMultistepScheduler,
}
