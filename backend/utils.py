import os
import time
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm


class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        if self.name:
            print(f"{self.name} took {elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {elapsed_time:.6f} seconds")


def download_file(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            content_length = int(response.headers.get("content-length", 0))
            progress = tqdm(total=content_length, unit="iB", unit_scale=True)
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress.update(len(chunk))
                    f.write(chunk)
            progress.close()
        else:
            print(f"Failed to download the file, status code: {response.status_code}")


def remove_none_fields(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: remove_none_fields(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_none_fields(elem) for elem in data]
    else:
        return data


def normalize_path(path):
    return path.replace("\\", "/")


def create_thumbnail(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    thumbnail_size = min(max_size, max(width, height))

    aspect_ratio = float(width) / float(height)
    if aspect_ratio > 1:
        new_width = thumbnail_size
        new_height = int(thumbnail_size / aspect_ratio)
    else:
        new_height = thumbnail_size
        new_width = int(thumbnail_size * aspect_ratio)

    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    thumbnail = Image.new("RGBA", (thumbnail_size, thumbnail_size), (0, 0, 0, 0))
    position = ((thumbnail_size - new_width) // 2, (thumbnail_size - new_height) // 2)
    thumbnail.paste(scaled_image, position)
    return thumbnail


def set_seed(seed):
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
