from collections import defaultdict

import os
import safetensors
import torch
from diffusers import DiffusionPipeline
from torch import Tensor

LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"

class LoraModel:
    layer_elems: dict[str, dict[str, Tensor]]

def load(path, device, dtype) -> LoraModel:
    model = LoraModel()

    _, ext = os.path.splitext(path)
    if ext == '.safetensors':
        state_dict = safetensors.torch.load_file(path)
    else:
        state_dict = torch.load(path, map_location='cpu')

    model.layer_elems = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        model.layer_elems[layer][elem] = value.to(device=device, dtype=dtype)

    return model

def apply(pipe: DiffusionPipeline, models: list[LoraModel], multipliers: list[float]):
    # Restore previous weights
    with torch.no_grad():
        for layer, weight_data in pipe.backup_weights.items():
            layer.weight.copy_(weight_data)
    pipe.backup_weights = {}

    for model, multiplier in zip(models, multipliers):
        for layer_name, elems in model.layer_elems.items():

            if "text" in layer_name:
                layer_infos = layer_name.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipe.text_encoder
            else:
                layer_infos = layer_name.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipe.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight']
            weight_down = elems['lora_down.weight']
            alpha = elems.get('alpha', None)
            if alpha is not None:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # save weight
            if curr_layer not in pipe.backup_weights:
                pipe.backup_weights[curr_layer] = curr_layer.weight.detach().clone().cpu()

            # update weight
            with torch.no_grad():
                if len(weight_up.shape) == 4:
                    updown = multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                else:
                    updown = multiplier * alpha * torch.mm(weight_up, weight_down)
            
            curr_layer.weight.data += updown
