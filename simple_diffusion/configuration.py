from dataclasses import dataclass

from diffusers import (DDIMScheduler, DDPMScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, SchedulerMixin, UniPCMultistepScheduler)
from processors import (CannyProcessor, DepthProcessor, LineartAnimeProcessor,
                        LineartCoarseProcessor, LineartProcessor,
                        MlsdProcessor, NormalBaeProcessor,
                        OpenposeFullProcessor, OpenposeProcessor, HedProcessor,
                        ProcessorBase, ScribbleHEDProcessor,
                        ScribblePIDIProcessor, SegProcessor, ShuffleProcessor,
                        SoftEdgeHEDProcessor, SoftEdgePIDIProcessor)
from PySide6.QtCore import QSize

APP_NAME = 'SimpleDiffusion'
APP_VERSION = 0.1

IMAGES_PATH = 'images'
THUMBNAILS_PATH = 'thumbnails'
MODELS_PATH = '.models'

ICON_SIZE = QSize(24, 24)

@dataclass
class ControlNetConfig:
    repo_id: str
    subfolder: str
    preprocessor: ProcessorBase

control_net_models: dict[str, ControlNetConfig] = {
    'Canny': ControlNetConfig('lllyasviel/control_v11p_sd15_canny', None, CannyProcessor),
    'Depth Midas': ControlNetConfig('lllyasviel/control_v11f1p_sd15_depth', None, DepthProcessor),
    # 'Inpaint': ControlNetConfig('lllyasviel/control_v11p_sd15_inpaint', None, None),
    'Instruct Pix2Pix': ControlNetConfig('lllyasviel/control_v11e_sd15_ip2p', None, None),
    'Lineart': ControlNetConfig('lllyasviel/control_v11p_sd15_lineart', None, LineartProcessor),
    'Lineart Coarse': ControlNetConfig('lllyasviel/control_v11p_sd15_lineart', None, LineartCoarseProcessor),
    'Lineart Anime': ControlNetConfig('lllyasviel/control_v11p_sd15s2_lineart_anime', None, LineartAnimeProcessor),
    'M-LSD Lines': ControlNetConfig('lllyasviel/control_v11p_sd15_mlsd', None, MlsdProcessor),
    'Normal BAE': ControlNetConfig('lllyasviel/control_v11p_sd15_normalbae', None, NormalBaeProcessor),
    'OpenPose': ControlNetConfig('lllyasviel/control_v11p_sd15_openpose', None, OpenposeProcessor),
    'OpenPose Full': ControlNetConfig('lllyasviel/control_v11p_sd15_openpose', None, OpenposeFullProcessor),
    'Scribble HED': ControlNetConfig('takuma104/control_v11', 'control_v11p_sd15_scribble', ScribbleHEDProcessor),
    'Scribble PIDI': ControlNetConfig('takuma104/control_v11', 'control_v11p_sd15_scribble', ScribblePIDIProcessor),
    'Segmentation': ControlNetConfig('lllyasviel/control_v11p_sd15_seg', None, SegProcessor),
    'Shuffle': ControlNetConfig('lllyasviel/control_v11e_sd15_shuffle', None, ShuffleProcessor), 
    'Soft Edge HED': ControlNetConfig('lllyasviel/control_v11p_sd15_softedge', None, SoftEdgeHEDProcessor),
    'Soft Edge PIDI': ControlNetConfig('lllyasviel/control_v11p_sd15_softedge', None, SoftEdgePIDIProcessor),
    # 'Tile': ControlNetConfig('lllyasviel/control_v11u_sd15_tile', None, None),
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
