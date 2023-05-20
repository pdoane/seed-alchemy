import os

from dataclasses import dataclass
import torch

from diffusers import (DDIMScheduler, DDPMScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, SchedulerMixin, UniPCMultistepScheduler)
from .processors import (CannyProcessor, DepthProcessor, LineartAnimeProcessor,
                        LineartCoarseProcessor, LineartProcessor,
                        MlsdProcessor, NormalBaeProcessor,
                        OpenposeFullProcessor, OpenposeProcessor, HedProcessor,
                        ProcessorBase, ScribbleHEDProcessor,
                        ScribblePIDIProcessor, SegProcessor, ShuffleProcessor,
                        SoftEdgeHEDProcessor, SoftEdgePIDIProcessor)
from PySide6.QtCore import QSize, QSettings

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
