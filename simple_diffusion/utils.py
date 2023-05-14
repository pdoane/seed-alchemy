import os
import re
import sys
import time
from dataclasses import MISSING, fields, is_dataclass
from typing import Callable

import requests
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFrame

from . import font_awesome as fa

if sys.platform == 'darwin':
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
    if sys.platform == 'darwin':
        url = NSURL.fileURLWithPath_(path)
        NSWorkspace.sharedWorkspace().activateFileViewerSelectingURLs_([url])

def recycle_file(path: str) -> None:
    if sys.platform == 'darwin':
        url = NSURL.fileURLWithPath_(path)
        NSWorkspace.sharedWorkspace().recycleURLs_completionHandler_([url], None)
    else:
        os.remove(full_path)

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

def create_thumbnail(image):
    width, height = image.size
    thumbnail_size = min(256, max(width, height))

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

def empty_qicon():
    global empty_icon
    if empty_icon is None:
        empty_pixmap = QPixmap(16, 16)
        empty_pixmap.fill(Qt.transparent)
        empty_icon = QIcon(empty_pixmap)
    return empty_icon

def create_fontawesome_icon(icon_code, size=16, color=Qt.white):
    app_instance = QApplication.instance()
    device_pixel_ratio = app_instance.devicePixelRatio()
    
    font = QFont(fa.font_family, size * device_pixel_ratio)
    pixmap = QPixmap(size * device_pixel_ratio, size * device_pixel_ratio)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setFont(font)

    if color:
        painter.setPen(color)

    painter.drawText(pixmap.rect(), Qt.AlignCenter, icon_code)
    painter.end()

    return QIcon(pixmap)

def horizontal_separator():
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    return separator

def pil_to_qimage(pil_image: Image.Image):
    data = pil_image.convert('RGBA').tobytes('raw', 'RGBA')
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    return qimage

def from_dict(dataclass_type, data: dict):
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")

    filtered_data = {}

    for field in fields(dataclass_type):
        if field.name in data and isinstance(data[field.name], field.type):
            filtered_data[field.name] = data[field.name]
        elif field.default != field.default_factory:
            filtered_data[field.name] = field.default
        elif field.default_factory != MISSING:
            filtered_data[field.name] = field.default_factory()
        else:
            raise ValueError(f"Missing value for field {field.name}")

    return dataclass_type(**filtered_data)

def set_current_data(widget, data):
    index = widget.findData(data)
    if index != -1:
        widget.setCurrentIndex(index)

def deserialize_string_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    else:
        return [str(value)]
