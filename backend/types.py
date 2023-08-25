from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelType(str, Enum):
    Checkpoint = "checkpoint"
    ControlNet = "controlnet"
    Lora = "lora"
    PromptGen = "promptgen"
    TextualInversion = "textual-inversion"
    Vae = "vae"


class BaseModelType(str, Enum):
    SD_1 = "sd-1"
    SD_2 = "sd-2"
    SDXL = "sdxl"
    SDXL_REFINER = "sdxl-refiner"


@dataclass
class ModelInfo:
    path: str
    local: bool
    type: ModelType
    base: Optional[BaseModelType] = None
    subfolder: Optional[str] = None
