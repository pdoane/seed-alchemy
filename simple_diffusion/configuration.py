from dataclasses import dataclass

from diffusers import (DDIMScheduler, DDPMScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, SchedulerMixin, UniPCMultistepScheduler)
from processors import (CannyProcessor, DepthProcessor, HedProcessor,
                        MlsdProcessor, NormalProcessor,
                        OpenposeProcessor, ProcessorBase,
                        ScribbleProcessor, SegProcessor)

APP_NAME = 'SimpleDiffusion'
APP_VERSION = 0.1
IMAGES_PATH = 'images'
THUMBNAILS_PATH = 'thumbnails'

@dataclass
class Img2ImgCondition:
    pass

@dataclass
class ControlNetCondition:
    preprocessor: ProcessorBase
    models: dict[str, str]

conditions = {
    'Image': Img2ImgCondition(),
    'Canny': ControlNetCondition(CannyProcessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-canny',
        'SD 2.1': 'thibaud/controlnet-sd21-canny-diffusers'
    }),
    'Depth': ControlNetCondition(DepthProcessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-depth',
        'SD 2.1': 'thibaud/controlnet-sd21-depth-diffusers'
    }),
    'Normal': ControlNetCondition(NormalProcessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-normal',
    }),
    'HED': ControlNetCondition(HedProcessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-hed',
        'SD 2.1': 'thibaud/controlnet-sd21-hed-diffusers'
    }),
    'M-LSD': ControlNetCondition(MlsdProcessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-mlsd',
    }),
    'Openpose': ControlNetCondition(OpenposeProcessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-openpose',
        'SD 2.1': 'thibaud/controlnet-sd21-openpose-diffusers'
    }),
    'Scribble': ControlNetCondition(ScribbleProcessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-scribble',
        'SD 2.1': 'thibaud/controlnet-sd21-scribble-diffusers'
    }),
    'Segmentation': ControlNetCondition(SegProcessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-seg',
    }),
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
