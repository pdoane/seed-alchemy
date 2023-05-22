import os
from dataclasses import dataclass

import torch
from diffusers import (DDIMScheduler, DDPMScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, SchedulerMixin, UniPCMultistepScheduler)
from PySide6.QtCore import QSettings, QSize

from .processors import (CannyProcessor, DepthProcessor, HedProcessor,
                         LineartAnimeProcessor, LineartCoarseProcessor,
                         LineartProcessor, MlsdProcessor, NormalBaeProcessor,
                         OpenposeFullProcessor, OpenposeProcessor,
                         ProcessorBase, ScribbleHEDProcessor,
                         ScribblePIDIProcessor, SegProcessor, ShuffleProcessor,
                         SoftEdgeHEDProcessor, SoftEdgePIDIProcessor)

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

controlnet_preprocessors: dict[str, ProcessorBase] = {
    'none': None,
    'canny': CannyProcessor,
    'depth_midas': DepthProcessor,
    'lineart_anime': LineartAnimeProcessor,
    'lineart_coarse': LineartCoarseProcessor,
    'lineart_realistic': LineartProcessor,
    'mlsd': MlsdProcessor,
    'normal_bae': NormalBaeProcessor,
    'openpose_full': OpenposeFullProcessor,
    'openpose': OpenposeProcessor,
    'scribble_hed': ScribbleHEDProcessor,
    'scribble_pidinet': ScribblePIDIProcessor,
    'seg_ofade20k': SegProcessor,
    'shuffle': ShuffleProcessor,
    'softedge_hed': SoftEdgeHEDProcessor,
    'softedge_pidinet': SoftEdgePIDIProcessor,
}

controlnet_models: list[str] = [
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
