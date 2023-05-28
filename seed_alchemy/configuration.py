import os

import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    SchedulerMixin,
    UniPCMultistepScheduler,
)
from PySide6.QtCore import QSettings, QSize

from . import utils


APP_NAME = "SeedAlchemy"
APP_VERSION = 0.1

IMAGES_PATH = "images"
THUMBNAILS_PATH = "thumbnails"
MODELS_PATH = ".models"

EMBEDDINGS_DIR = "embeddings"
LORA_DIR = "lora"
STABLE_DIFFUSION_DIR = "stable_diffusion"

if torch.cuda.is_available():
    torch_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    torch_device = torch.device("mps")
else:
    torch_device = torch.device("cpu")
torch_dtype = torch.float32

resources_path: str
local_models_path: str
textual_inversions: dict[str, str]
loras: dict[str, str]
stable_diffusion_models: dict[str, str]

ICON_SIZE = QSize(24, 24)
font_scale_factor = 1.0


schedulers: dict[str, SchedulerMixin] = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "deis_multi": DEISMultistepScheduler,
    # 'dpm_multi': DPMSolverMultistepScheduler,
    # 'dpm': DPMSolverSinglestepScheduler,
    # 'k_dpm_2': KDPM2DiscreteScheduler,
    # 'k_dpm_2_a': KDPM2AncestralDiscreteScheduler,
    "k_euler": EulerDiscreteScheduler,
    "k_euler_a": EulerAncestralDiscreteScheduler,
    "k_heun": HeunDiscreteScheduler,
    "k_lms": LMSDiscreteScheduler,
    "pndm": PNDMScheduler,
    "uni_pc": UniPCMultistepScheduler,
}


def load_from_settings(settings: QSettings):
    global torch_dtype
    torch_dtype = torch.float32 if settings.value("float32", type=bool) else torch.float16

    global local_models_path
    local_models_path = settings.value("local_models_path")

    global textual_inversions
    textual_inversions = {}

    embeddings_path = os.path.join(local_models_path, EMBEDDINGS_DIR)
    if os.path.exists(embeddings_path):
        for entry in sorted(os.listdir(embeddings_path)):
            if entry == ".DS_Store":
                continue
            entry_path = os.path.join(embeddings_path, entry)
            if not os.path.isfile(entry_path):
                continue

            base_name, _ = os.path.splitext(entry)
            textual_inversions[base_name] = os.path.join(local_models_path, EMBEDDINGS_DIR, entry)

    global loras
    loras = {}

    loras_path = os.path.join(local_models_path, LORA_DIR)
    if os.path.exists(loras_path):
        for entry in sorted(os.listdir(loras_path)):
            if entry == ".DS_Store":
                continue
            entry_path = os.path.join(loras_path, entry)
            if not os.path.isfile(entry_path):
                continue

            base_name, _ = os.path.splitext(entry)
            loras[base_name] = os.path.join(local_models_path, LORA_DIR, entry)

    global stable_diffusion_models
    stable_diffusion_models = {}

    stable_diffusion_path = os.path.join(local_models_path, STABLE_DIFFUSION_DIR)
    if os.path.exists(stable_diffusion_path):
        for entry in sorted(os.listdir(stable_diffusion_path)):
            if entry == ".DS_Store":
                continue
            entry_path = os.path.join(stable_diffusion_path, entry)
            if not os.path.isdir(entry_path):
                continue

            base_name = os.path.basename(entry)
            stable_diffusion_models[base_name] = os.path.join(local_models_path, STABLE_DIFFUSION_DIR, entry)

    for repo_id in utils.deserialize_string_list(settings.value("huggingface_models")):
        base_name = os.path.basename(repo_id)
        stable_diffusion_models[base_name] = repo_id


def set_resources_path(path):
    global resources_path
    resources_path = path


def get_resource_path(relative_path) -> str:
    return os.path.join(resources_path, relative_path)
