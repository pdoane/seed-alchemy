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
                         MediapipeFaceProcessor, MlsdProcessor,
                         NormalBaeProcessor, NormalMidasProcessor,
                         OpenposeFaceOnlyProcessor, OpenposeFaceProcessor,
                         OpenposeFullProcessor, OpenposeHandProcessor,
                         OpenposeProcessor, ProcessorBase,
                         ScribbleHEDProcessor, ScribblePIDIProcessor,
                         ScribbleXDoGProcessor, SegProcessor, ShuffleProcessor,
                         SoftEdgeHEDProcessor, SoftEdgeHEDSafeProcessor,
                         SoftEdgePIDIProcessor, SoftEdgePIDISafeProcessor)

APP_NAME = 'SeedAlchemy'
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
    'mediapipe_face': MediapipeFaceProcessor,
    'mlsd': MlsdProcessor,
    'normal_bae': NormalBaeProcessor,
    'normal_midas': NormalMidasProcessor,
    'openpose': OpenposeProcessor,
    'openpose_face': OpenposeFaceProcessor,
    'openpose_faceonly': OpenposeFaceOnlyProcessor,
    'openpose_full': OpenposeFullProcessor,
    'openpose_hand': OpenposeHandProcessor,
    'scribble_hed': ScribbleHEDProcessor,
    'scribble_pidinet': ScribblePIDIProcessor,
    'scribble_xdog': ScribbleXDoGProcessor,
    'seg_ofade20k': SegProcessor,
    'shuffle': ShuffleProcessor,
    'softedge_hed': SoftEdgeHEDProcessor,
    'softedge_hedsafe': SoftEdgeHEDSafeProcessor,
    'softedge_pidinet': SoftEdgePIDIProcessor,
    'softedge_pidsafe': SoftEdgePIDISafeProcessor,
}

control_net_preprocessors_to_models: dict[str, list[str]] = {
    'none': [],
    'canny': ['control_v11p_sd15_canny','control_sd15_canny'],
    'depth_midas': ['control_v11f1p_sd15_depth','control_sd15_depth'],
    'depth_zoe': ['control_v11f1p_sd15_depth'],
    'lineart_anime': ['control_v11p_sd15s2_lineart_anime'],
    'lineart_coarse': ['control_v11p_sd15_lineart'],
    'lineart_realistic': ['control_v11p_sd15_lineart'],
    'mediapipe_face': ['control_v2p_sd15_mediapipe_face'],
    'mlsd': ['control_v11p_sd15_mlsd','control_sd15_mlsd'],
    'normal_bae': ['control_v11p_sd15_normalbae'],
    'normal_midas': ['control_sd15_normal'],
    'openpose': ['control_v11p_sd15_openpose','control_sd15_openpose'],
    'openpose_face': ['control_v11p_sd15_openpose'],
    'openpose_faceonly': ['control_v11p_sd15_openpose'],
    'openpose_full': ['control_v11p_sd15_openpose'],
    'openpose_hand': ['control_v11p_sd15_openpose'],
    'scribble_hed': ['control_v11p_sd15_scribble','control_sd15_scribble'],
    'scribble_pidinet': ['control_v11p_sd15_scribble','control_sd15_scribble'],
    'scribble_xdog': ['control_v11p_sd15_scribble','control_sd15_scribble'],
    'seg_ofade20k': ['control_v11p_sd15_seg','control_sd15_seg'],
    'shuffle': ['control_v11e_sd15_shuffle'],
    'softedge_hed': ['control_v11p_sd15_softedge','control_sd15_hed'],
    'softedge_hedsafe': ['control_v11p_sd15_softedge','control_sd15_hed'],
    'softedge_pidinet': ['control_v11p_sd15_softedge','control_sd15_hed'],
    'softedge_pidsafe': ['control_v11p_sd15_softedge','control_sd15_hed'],
 }

@dataclass
class ControlNetModel:
    repo_id: str
    subfolder: str = None

control_net_models: dict[str, ControlNetModel] = {
    'control_sd15_canny': ControlNetModel(repo_id='lllyasviel/sd-controlnet-canny'),
    'control_sd15_depth': ControlNetModel(repo_id='lllyasviel/sd-controlnet-depth'),
    'control_sd15_hed': ControlNetModel(repo_id='lllyasviel/sd-controlnet-hed'),
    'control_sd15_mlsd': ControlNetModel(repo_id='lllyasviel/sd-controlnet-mlsd'),
    'control_sd15_normal': ControlNetModel(repo_id='lllyasviel/sd-controlnet-normal'),
    'control_sd15_openpose': ControlNetModel(repo_id='lllyasviel/sd-controlnet-openpose'),
    'control_sd15_scribble': ControlNetModel(repo_id='lllyasviel/sd-controlnet-scribble'),
    'control_sd15_seg': ControlNetModel(repo_id='lllyasviel/sd-controlnet-seg'),

    'control_v11p_sd15_canny': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_canny'),
    'control_v11f1p_sd15_depth': ControlNetModel(repo_id='lllyasviel/control_v11f1p_sd15_depth'),
    #'control_v11p_sd15_inpaint': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_inpaint'),
    'control_v11e_sd15_ip2p': ControlNetModel(repo_id='lllyasviel/control_v11e_sd15_ip2p'),
    'control_v11p_sd15_lineart': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_linear  t'),
    'control_v11p_sd15s2_lineart_anime': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15s2_lineart_anime'),
    'control_v11p_sd15_mlsd': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_mlsd'),
    'control_v11p_sd15_normalbae': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_normalbae'),
    'control_v11p_sd15_openpose': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_openpose'),
    'control_v11p_sd15_scribble': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_scribble'),
    'control_v11p_sd15_seg': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_seg'),
    'control_v11e_sd15_shuffle': ControlNetModel(repo_id='lllyasviel/control_v11e_sd15_shuffle'),
    'control_v11p_sd15_softedge': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_softedge'),
    #'control_v11u_sd15_tile': ControlNetModel(repo_id='lllyasviel/control_v11u_sd15_tile'),

    'control_v2p_sd15_mediapipe_face': ControlNetModel(repo_id='CrucibleAI/ControlNetMediaPipeFace', subfolder='diffusion_sd15')
}

control_net_v10_models: list[str] = [
    'control_sd15_canny',
    'control_sd15_depth',
    'control_sd15_hed',
    'control_sd15_mlsd',
    'control_sd15_normal',
    'control_sd15_openpose',
    'control_sd15_scribble',
    'control_sd15_seg',
]

control_net_v11_models: list[str] = [
    'control_v11p_sd15_canny',
    'control_v11f1p_sd15_depth',
    #'control_v11p_sd15_inpaint',
    'control_v11e_sd15_ip2p',
    'control_v11p_sd15_lineart',
    'control_v11p_sd15s2_lineart_anime',
    'control_v11p_sd15_mlsd',
    'control_v11p_sd15_normalbae',
    'control_v11p_sd15_openpose',
    'control_v11p_sd15_scribble',
    'control_v11p_sd15_seg',
    'control_v11e_sd15_shuffle',
    'control_v11p_sd15_softedge',
    #'control_v11u_sd15_tile',
]

control_net_mediapipe_v2_models: list[str] = {
    'control_v2p_sd15_mediapipe_face'
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
