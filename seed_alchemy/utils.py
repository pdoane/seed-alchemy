import os
import re
import subprocess
import sys
import time
from typing import Any, Callable

import requests
import send2trash
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPixmap


if sys.platform == "darwin":
    from AppKit import NSURL, NSWorkspace

empty_icon: QIcon = None


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


def reveal_in_finder(path: str) -> None:
    if sys.platform == "darwin":
        url = NSURL.fileURLWithPath_(path)
        NSWorkspace.sharedWorkspace().activateFileViewerSelectingURLs_([url])
    elif sys.platform == "win32":
        subprocess.run(["explorer", "/select,", path])


def recycle_file(path: str) -> None:
    send2trash.send2trash(path)


def download_file(url: str, output_path: str) -> None:
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"Failed to download the file, status code: {response.status_code}")


def next_image_id(path: str) -> int:
    index = 0
    for image_file in os.listdir(path):
        match = re.match(r"(\d+)\.png", image_file)
        if match:
            index = max(index, int(match.group(1)))
    return index + 1


def retry_on_failure(operation: Callable, max_retries=10, initial_delay=0.1, backoff_factor=1.1):
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

    return None


def create_thumbnail(image, max_size):
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


def empty_qicon():
    global empty_icon
    if empty_icon is None:
        empty_pixmap = QPixmap(16, 16)
        empty_pixmap.fill(Qt.transparent)
        empty_icon = QIcon(empty_pixmap)
    return empty_icon


def pil_to_qimage(pil_image: Image.Image):
    data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    return qimage


def set_current_data(widget, data):
    index = widget.findData(data)
    if index != -1:
        widget.setCurrentIndex(index)


def deserialize_string_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    else:
        return [str(value)]


def list_get(l, index, default=None):
    try:
        return l[index]
    except IndexError:
        return default


def remove_none_fields(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: remove_none_fields(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_none_fields(elem) for elem in data]
    else:
        return data
