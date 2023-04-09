import os
import re
import sys
import time
from typing import Callable

import requests
from PIL import Image

if sys.platform == 'darwin':
    from AppKit import NSURL, NSWorkspace


class ChangeDirectory:
    def __init__(self, dir) -> None:
        self.dir = dir
        self.orig_dir = os.getcwd()

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.orig_dir)

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

def resource_path(relative_path) -> str:
    return os.path.join('simple_diffusion/resources', relative_path)

def reveal_in_finder(file_path: str) -> None:
    if sys.platform == 'darwin':
        file_url = NSURL.fileURLWithPath_(file_path)
        NSWorkspace.sharedWorkspace().activateFileViewerSelectingURLs_([file_url])

def download_file(url: str, output_path: str) -> None:
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f'Failed to download the file, status code: {response.status_code}')

def next_image_id(dir: str) -> int:
    id = 0
    for image_file in os.listdir(dir):
        match = re.match(r'(\d+)\.png', image_file)
        if match:
            id = max(id, int(match.group(1)))
    return id + 1

def retry_on_failure(operation: Callable, max_retries=10, initial_delay=0.1, backoff_factor=2):
    current_retry = 0

    while current_retry < max_retries:
        try:
            result = operation()
            return result
        except Exception as e:
            current_retry += 1
            if current_retry == max_retries:
                raise e

            delay = initial_delay * (backoff_factor ** (current_retry - 1))
            time.sleep(delay)

def create_thumbnail(image):
    width, height = image.size
    thumbnail_size = max(width, height)

    aspect_ratio = float(width) / float(height)
    if aspect_ratio > 1:
        new_width = thumbnail_size
        new_height = int(thumbnail_size / aspect_ratio)
    else:
        new_height = thumbnail_size
        new_width = int(thumbnail_size * aspect_ratio)

    scaled_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    thumbnail = Image.new('RGBA', (thumbnail_size, thumbnail_size), (0, 0, 0, 0))
    position = ((thumbnail_size - new_width) // 2, (thumbnail_size - new_height) // 2)
    thumbnail.paste(scaled_image, position)
    return thumbnail
