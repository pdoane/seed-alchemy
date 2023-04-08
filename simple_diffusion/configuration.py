from dataclasses import dataclass

from diffusers import (DDIMScheduler, DDPMScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, SchedulerMixin, UniPCMultistepScheduler)

from preprocessors import (CannyPreprocessor, DepthPreprocessor,
                           HedPreprocessor, MlsdPreprocessor,
                           NormalPreprocessor, OpenposePreprocessor,
                           PreprocessorBase, ScribblePreprocessor,
                           SegPreprocessor)

APP_NAME = 'SimpleDiffusion'
APP_VERSION = 0.1
IMAGES_PATH = 'images'
THUMBNAILS_PATH = 'thumbnails'

@dataclass
class Img2ImgCondition:
    pass

@dataclass
class ControlNetCondition:
    preprocessor: PreprocessorBase
    models: dict[str, str]

conditions = {
    'Image': Img2ImgCondition(),
    'Canny': ControlNetCondition(CannyPreprocessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-canny',
        'SD 2.1': 'thibaud/controlnet-sd21-canny-diffusers'
    }),
    'Depth': ControlNetCondition(DepthPreprocessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-depth',
        'SD 2.1': 'thibaud/controlnet-sd21-depth-diffusers'
    }),
    'Normal': ControlNetCondition(NormalPreprocessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-normal',
    }),
    'HED': ControlNetCondition(HedPreprocessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-hed',
        'SD 2.1': 'thibaud/controlnet-sd21-hed-diffusers'
    }),
    'M-LSD': ControlNetCondition(MlsdPreprocessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-mlsd',
    }),
    'Openpose': ControlNetCondition(OpenposePreprocessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-openpose',
        'SD 2.1': 'thibaud/controlnet-sd21-openpose-diffusers'
    }),
    'Scribble': ControlNetCondition(ScribblePreprocessor, {
        'SD 1.5': 'lllyasviel/sd-controlnet-scribble',
        'SD 2.1': 'thibaud/controlnet-sd21-scribble-diffusers'
    }),
    'Segmentation': ControlNetCondition(SegPreprocessor, {
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
