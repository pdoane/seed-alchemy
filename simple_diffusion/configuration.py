from dataclasses import dataclass

from diffusers import (DDIMScheduler, DDPMScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, SchedulerMixin, UniPCMultistepScheduler)
from processors import (CannyProcessor, DepthProcessor, LineartAnimeProcessor,
                        LineartCoarseProcessor, LineartProcessor,
                        MlsdProcessor, NormalBaeProcessor,
                        OpenposeFullProcessor, OpenposeProcessor,
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
class ControlNetModel:
    repo_id: str
    subfolder: str
    preprocessor: ProcessorBase

control_net_models: dict[str, ControlNetModel] = {
    'Canny Edges': ControlNetModel('lllyasviel/control_v11p_sd15_canny', None, CannyProcessor),
    'Depth Map': ControlNetModel('lllyasviel/control_v11p_sd15_depth', None, DepthProcessor),
    # 'Inpaint': ControlNetModel('lllyasviel/control_v11p_sd15_inpaint', None, None),
    'Instruct Pix2Pix': ControlNetModel('lllyasviel/control_v11e_sd15_ip2p', None, None),
    'Lineart': ControlNetModel('lllyasviel/control_v11p_sd15_lineart', None, LineartProcessor),
    'Lineart Coarse': ControlNetModel('lllyasviel/control_v11p_sd15_lineart', None, LineartCoarseProcessor),
    'Lineart Anime': ControlNetModel('lllyasviel/control_v11p_sd15s2_lineart_anime', None, LineartAnimeProcessor),
    'M-LSD Lines': ControlNetModel('lllyasviel/control_v11p_sd15_mlsd', None, MlsdProcessor),
    'Normal Map': ControlNetModel('lllyasviel/control_v11p_sd15_normalbae', None, NormalBaeProcessor),
    'OpenPose': ControlNetModel('lllyasviel/control_v11p_sd15_openpose', None, OpenposeProcessor),
    'OpenPose Full': ControlNetModel('lllyasviel/control_v11p_sd15_openpose', None, OpenposeFullProcessor),
    'Scribble HED': ControlNetModel('takuma104/control_v11', 'control_v11p_sd15_scribble', ScribbleHEDProcessor),
    'Scribble PIDI': ControlNetModel('takuma104/control_v11', 'control_v11p_sd15_scribble', ScribblePIDIProcessor),
    'Segmentation': ControlNetModel('lllyasviel/control_v11p_sd15_seg', None, SegProcessor),
    'Shuffle': ControlNetModel('lllyasviel/control_v11e_sd15_shuffle', None, ShuffleProcessor), 
    'Soft Edge HED': ControlNetModel('lllyasviel/control_v11p_sd15_softedge', None, SoftEdgeHEDProcessor),
    'Soft Edge PIDI': ControlNetModel('lllyasviel/control_v11p_sd15_softedge', None, SoftEdgePIDIProcessor),
    # 'Tile': ControlNetModel('lllyasviel/control_v11u_sd15_tile', None, None),
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
