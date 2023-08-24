import sys
import torch


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def default_dtype():
    if sys.platform == "darwin":
        return torch.float32        # TODO - user setting
    else:
        return torch.float16