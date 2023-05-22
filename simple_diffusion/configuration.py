import os
from dataclasses import dataclass

import torch
from diffusers import (DDIMScheduler, DDPMScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, SchedulerMixin, UniPCMultistepScheduler)
from PySide6.QtCore import QSettings, QSize

from .processors import (CannyProcessor, DepthMidasProcessor,
                         DepthZoeProcessor, LineartAnimeProcessor,
                         LineartCoarseProcessor, LineartProcessor,
                         MlsdProcessor, NormalBaeProcessor,
                         NormalMidasProcessor, OpenposeFullProcessor,
                         OpenposeProcessor, ProcessorBase,
                         ScribbleHEDProcessor, ScribblePIDIProcessor,
                         SegProcessor, ShuffleProcessor, SoftEdgeHEDProcessor,
                         SoftEdgeHEDSafeProcessor, SoftEdgePIDIProcessor,
                         SoftEdgePIDISafeProcessor)

APP_NAME = 'SimpleDiffusion'
APP_VERSION = 0.1

IMAGES_PATH = 'images'
THUMBNAILS_PATH = 'thumbnails'
MODELS_PATH = '.models'

EMBEDDINGS_DIR = 'embeddings'
LORA_DIR = 'lora'
STABLE_DIFFUSION_DIR = 'stable_diffusion'

if torch.cuda.is_available():
    torch_device = 'cuda'
elif torch.backends.mps.is_available():
    torch_device = 'mps'
else:
    torch_device = 'cpu'
torch_dtype = torch.float32

resources_path: str
local_models_path: str
known_embeddings: list[str] = []
known_loras: list[str] = []
lora_dict: dict[str, str] = {}
known_stable_diffusion_models: list[str] = []

ICON_SIZE = QSize(24, 24)
font_scale_factor = 1.0

control_net_preprocessors: dict[str, ProcessorBase] = {
    'none': None,
    'canny': CannyProcessor,
    'depth_midas': DepthMidasProcessor,
    'depth_zoe': DepthZoeProcessor,
    'lineart_anime': LineartAnimeProcessor,
    'lineart_coarse': LineartCoarseProcessor,
    'lineart_realistic': LineartProcessor,
    'mlsd': MlsdProcessor,
    'normal_bae': NormalBaeProcessor,
    'normal_midas': NormalMidasProcessor,
    'openpose_full': OpenposeFullProcessor,
    'openpose': OpenposeProcessor,
    'scribble_hed': ScribbleHEDProcessor,
    'scribble_pidinet': ScribblePIDIProcessor,
    'seg_ofade20k': SegProcessor,
    'shuffle': ShuffleProcessor,
    'softedge_hed': SoftEdgeHEDProcessor,
    'softedge_hedsafe': SoftEdgeHEDSafeProcessor,
    'softedge_pidinet': SoftEdgePIDIProcessor,
    'softedge_pidsafe': SoftEdgePIDISafeProcessor,
}

@dataclass
class ControlNetParameter:
    type: type
    name: str
    min: float
    max: float
    value: float
    step: float = 1

control_net_parameters: dict[str, list[ControlNetParameter]] = {
    'canny': [
        ControlNetParameter(type=int, name='Low', min=1, max=255, value=100),
        ControlNetParameter(type=int, name='High', min=1, max=255, value=200),
    ],
    'mlsd': [
        ControlNetParameter(type=float, name='Value', min=0.01, max=2.0, value=0.1, step=0.01),
        ControlNetParameter(type=float, name='Distance', min=0.01, max=20.0, value=0.1, step=0.01),
    ],
    'normal_midas': [
        ControlNetParameter(type=float, name='Background', min=0.0, max=1.0, value=0.4, step=0.01)
    ],
}

control_net_preprocessors_to_models: dict[str, list[str]] = {
    'none': [],
    'canny': ['lllyasviel/control_v11p_sd15_canny','lllyasviel/sd-controlnet-canny'],
    'depth_midas': ['lllyasviel/control_v11f1p_sd15_depth','lllyasviel/sd-controlnet-depth'],
    'depth_zoe': ['lllyasviel/control_v11f1p_sd15_depth'],
    'lineart_anime': ['lllyasviel/control_v11p_sd15s2_lineart_anime'],
    'lineart_coarse': ['lllyasviel/control_v11p_sd15_lineart'],
    'lineart_realistic': ['lllyasviel/control_v11p_sd15_lineart'],
    'mlsd': ['lllyasviel/control_v11p_sd15_mlsd','lllyasviel/sd-controlnet-mlsd'],
    'normal_bae': ['lllyasviel/control_v11p_sd15_normalbae'],
    'normal_midas': ['lllyasviel/sd-controlnet-normal'],
    'openpose_full': ['lllyasviel/control_v11p_sd15_openpose'],
    'openpose': ['lllyasviel/control_v11p_sd15_openpose','lllyasviel/sd-controlnet-openpose'],
    'scribble_hed': ['lllyasviel/control_v11p_sd15_scribble','lllyasviel/sd-controlnet-scribble'],
    'scribble_pidinet': ['lllyasviel/control_v11p_sd15_scribble','lllyasviel/sd-controlnet-scribble'],
    'seg_ofade20k': ['lllyasviel/control_v11p_sd15_seg','lllyasviel/sd-controlnet-seg'],
    'shuffle': ['lllyasviel/control_v11e_sd15_shuffle'],
    'softedge_hed': ['lllyasviel/control_v11p_sd15_softedge','lllyasviel/sd-controlnet-hed'],
    'softedge_hedsafe': ['lllyasviel/control_v11p_sd15_softedge','lllyasviel/sd-controlnet-hed'],
    'softedge_pidinet': ['lllyasviel/control_v11p_sd15_softedge','lllyasviel/sd-controlnet-hed'],
    'softedge_pidsafe': ['lllyasviel/control_v11p_sd15_softedge','lllyasviel/sd-controlnet-hed'],
 }

control_net10_models: list[str] = [
    'lllyasviel/sd-controlnet-canny',
    'lllyasviel/sd-controlnet-depth',
    'lllyasviel/sd-controlnet-hed',
    'lllyasviel/sd-controlnet-mlsd',
    'lllyasviel/sd-controlnet-normal',
    'lllyasviel/sd-controlnet-openpose',
    'lllyasviel/sd-controlnet-scribble',
    'lllyasviel/sd-controlnet-seg',
]

control_net11_models: list[str] = [
    'lllyasviel/control_v11p_sd15_canny',
    'lllyasviel/control_v11f1p_sd15_depth',
    #'lllyasviel/control_v11p_sd15_inpaint',
    'lllyasviel/control_v11e_sd15_ip2p',
    'lllyasviel/control_v11p_sd15_lineart',
    'lllyasviel/control_v11p_sd15s2_lineart_anime',
    'lllyasviel/control_v11p_sd15_mlsd',
    'lllyasviel/control_v11p_sd15_normalbae',
    'lllyasviel/control_v11p_sd15_openpose',
    'lllyasviel/control_v11p_sd15_scribble',
    'lllyasviel/control_v11p_sd15_seg',
    'lllyasviel/control_v11e_sd15_shuffle',
    'lllyasviel/control_v11p_sd15_softedge',
    #'lllyasviel/control_v11u_sd15_tile',
]

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

def load_from_settings(settings: QSettings):
    global torch_dtype
    torch_dtype = torch.float32 if settings.value('float32', type=bool) else torch.float16

    global local_models_path
    local_models_path = settings.value('local_models_path')

    global known_embeddings
    known_embeddings = []

    embeddings_path = os.path.join(local_models_path, EMBEDDINGS_DIR)
    if os.path.exists(embeddings_path):
        for entry in sorted(os.listdir(embeddings_path)):
            if entry == '.DS_Store':
                continue
            entry_path = os.path.join(embeddings_path, entry)
            if not os.path.isfile(entry_path):
                continue

            known_embeddings.append(entry)

    global known_loras
    known_loras = []

    loras_path = os.path.join(local_models_path, LORA_DIR)
    if os.path.exists(loras_path):
        for entry in sorted(os.listdir(loras_path)):
            if entry == '.DS_Store':
                continue
            entry_path = os.path.join(loras_path, entry)
            if not os.path.isfile(entry_path):
                continue

            known_loras.append(entry)

            base_name,_ = os.path.splitext(entry)
            lora_dict[base_name] = entry
    
    global known_stable_diffusion_models
    known_stable_diffusion_models = []

    stable_diffusion_path = os.path.join(local_models_path, STABLE_DIFFUSION_DIR)
    if os.path.exists(stable_diffusion_path):
        for entry in sorted(os.listdir(stable_diffusion_path)):
            if entry == '.DS_Store':
                continue
            entry_path = os.path.join(stable_diffusion_path, entry)
            if not os.path.isdir(entry_path):
                continue

            known_stable_diffusion_models.append(entry)

def get_embedding_path(str):
    return os.path.join(local_models_path, EMBEDDINGS_DIR, str)

def get_lora_path(str):
    return os.path.join(local_models_path, LORA_DIR, lora_dict[str])

def get_stable_diffusion_model_path(str):
    return os.path.join(local_models_path, STABLE_DIFFUSION_DIR, str)

def set_resources_path(str):
    global resources_path
    resources_path = str

def get_resource_path(relative_path) -> str:
    return os.path.join(resources_path, relative_path)
