import os
import re
import uuid
from typing import Optional

from pydantic import BaseSettings, Field

from . import control_net_registry
from .types import ModelType, BaseModelType, ModelInfo


class Settings(BaseSettings):
    users: list[str] = Field(default_factory=lambda: ["default"], split=",")
    cache_path: str = "~/.cache"
    storage_path: str = "storage"
    models_path: Optional[str] = None
    huggingface_models: list[str] = Field(
        default_factory=lambda: [
            "checkpoint:sd-1:runwayml/stable-diffusion-v1-5",
            "checkpoint:sd-2:stabilityai/stable-diffusion-2-1",
            "checkpoint:sdxl:stabilityai/stable-diffusion-xl-base-1.0",
            "promptgen::AUTOMATIC/promptgen-lexart",
        ],
        split=",",
    )
    install_control_net_v10: bool = False
    install_control_net_v11: bool = True
    install_control_net_mediapipe_v2: bool = True

    def __str__(self):
        return "\n".join(f"{key}={value}" for key, value in self.dict().items())


settings: Settings
models: dict[str, ModelInfo] = {}  # Different namespace for each base model?
promptgen_models: dict[str, str] = {}


def is_valid_diffusers_model(path: str):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "model_index.json")):
        return True
    return False


def is_valid_single_file(path: str):
    if os.path.isfile(path):
        _, ext = os.path.splitext(path)
        return ext == ".safetensors" or ext == ".pt" or ext == ".ckpt" or ext == ".pth"
    return False


def safe_list_dir(path: str) -> list[str]:
    if os.path.isdir(path):
        return os.listdir(path)
    else:
        return []


def add_hf_model(type, base, path, name):
    components = path.split("/")
    if len(components) == 2:
        repo_id = "/".join(components)
        subfolder = None
    else:
        repo_id = "/".join(components[:2])
        subfolder = "/".join(components[2:])

    info = ModelInfo(path=repo_id, subfolder=subfolder, local=False, type=type, base=base)
    models[name] = info


def add_hf_models(entries):
    for entry in entries:
        add_hf_model(entry[0], entry[1], entry[2], entry[3])


def load_settings(root_dir: str):
    global settings
    if root_dir:
        root_dir = os.path.abspath(os.path.expanduser(root_dir))
        env_file = os.path.join(root_dir, ".env")
    else:
        root_dir = os.path.abspath(".")
        env_file = ".env"
    settings = Settings(_env_file=env_file)

    def resolve_path(path):
        if path is None:
            return None
        return os.path.join(root_dir, os.path.expanduser(path))

    settings.cache_path = resolve_path(settings.cache_path)
    settings.storage_path = resolve_path(settings.storage_path)
    settings.models_path = resolve_path(settings.models_path)

    # Diffusion Models
    models.clear()
    if settings.models_path:
        for base in [BaseModelType.SD_1, BaseModelType.SD_2, BaseModelType.SDXL, BaseModelType.SDXL_REFINER]:
            base_path = os.path.join(settings.models_path, base)
            if os.path.exists(base_path):
                for type in [
                    ModelType.Checkpoint,
                    ModelType.ControlNet,
                    ModelType.Lora,
                    ModelType.TextualInversion,
                    ModelType.Vae,
                ]:
                    type_path = os.path.join(base_path, type)
                    for name in safe_list_dir(type_path):
                        path = os.path.join(type_path, name)
                        info = ModelInfo(path=path, local=True, type=type, base=base)
                        if is_valid_diffusers_model(path):
                            models[name] = info
                        elif is_valid_single_file(path):
                            base_name, _ = os.path.splitext(name)
                            models[base_name] = info

    if settings.install_control_net_v10:
        add_hf_models(control_net_registry.v10_models)
    if settings.install_control_net_v11:
        add_hf_models(control_net_registry.v11_models)
    if settings.install_control_net_mediapipe_v2:
        add_hf_models(control_net_registry.mediapipe_v2_models)
    for model in settings.huggingface_models:
        components = model.split(":")
        if len(components) != 3 and len(components) != 4:
            raise ValueError("Huggingface models must be in the form type:base:repo_id/subfolder[:name]")
        path_components = components[2].split("/")
        if len(path_components) < 2:
            raise ValueError("Huggingface models must be in the form type:base:repo_id/subfolder[:name]")

        if len(components) == 4:
            name = components[3]
        else:
            name = "/".join(path_components[1:])

        add_hf_model(type=components[0], base=components[1], path=components[2], name=name)


def get_settings_path(user: str):
    return os.path.join(settings.storage_path, user, "settings.json")


def get_images_path(user: str):
    return os.path.join(settings.storage_path, user, "images")


def get_image_path(user: str, path: str):
    return os.path.join(settings.storage_path, user, "images", path)


def get_thumbnail_path(user: str, path: str):
    return os.path.join(settings.storage_path, user, "thumbnails", path)


def get_cache_path(subfolder: str, path: Optional[str]):
    cache_dir = os.path.expanduser(settings.cache_path)
    if path:
        return os.path.join(cache_dir, subfolder, path)
    else:
        return os.path.join(cache_dir, subfolder)


def generate_output_path(user: str, dir: str) -> int:
    full_path = get_image_path(user, dir)
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    index = 0
    for image_file in os.listdir(full_path):
        match = re.match(r"(\d+)\.[0-9a-f]+\.png", image_file)
        if match:
            index = max(index, int(match.group(1)))
        match = re.match(r"(\d+)\.png", image_file)
        if match:
            index = max(index, int(match.group(1)))

    index += 1

    short_uuid = str(uuid.uuid4())[:8]
    return os.path.join(dir, "{:05d}.{:s}.png".format(index, short_uuid))
