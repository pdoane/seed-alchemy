from __future__ import annotations  # Needed for Python 3.7-3.9

import os

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import configuration
from .image_metadata import ImageMetadata
from .thumbnail_loader import ThumbnailLoader
from .thumbnail_model import ThumbnailModel


class SourceImageWidget(QWidget):
    def __init__(self, thumbnail_loader: ThumbnailLoader, thumbnail_model: ThumbnailModel, parent=None):
        super().__init__(parent)

        self.thumbnail_loader = thumbnail_loader
        self.thumbnail_model = thumbnail_model
        self.label: QLabel = None
        self.line_edit: QLineEdit = None
        self.context_menu: QMenu = None
        self.previous_image_path: str = None
        self.init_ui()

    def init_ui(self):
        source_label = QLabel("Source")
        source_label.setAlignment(Qt.AlignCenter)

        self.label = QLabel()
        self.label.setFrameStyle(QFrame.Box)
        self.label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label.setFixedSize(96, 96)

        self.line_edit = QLineEdit()
        self.line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.line_edit.textChanged.connect(self.on_text_changed)
        self.line_edit.setPlaceholderText("Image Path")

        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(2)
        vlayout.addStretch(1)
        vlayout.addWidget(source_label)
        vlayout.addWidget(self.line_edit)
        vlayout.addStretch(2)

        hlayout = QHBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(2)
        hlayout.addLayout(vlayout)
        hlayout.addWidget(self.label)

    def destroy(self):
        image_path = self.line_edit.text()
        self.thumbnail_model.remove_reference(image_path)

    def on_text_changed(self):
        image_path = self.line_edit.text()
        self.thumbnail_loader.get(image_path, 96, self.on_thumbnail_loaded)

        if self.previous_image_path:
            self.thumbnail_model.remove_reference(self.previous_image_path)
        self.thumbnail_model.add_reference(image_path)
        self.previous_image_path = image_path

    def get_metadata(self):
        image_path = self.line_edit.text()
        if image_path:
            full_path = os.path.join(configuration.IMAGES_PATH, image_path)
            with Image.open(full_path) as image:
                metadata = ImageMetadata()
                metadata.path = image_path
                metadata.load_from_image(image)
                return metadata
        return None

    def on_thumbnail_loaded(self, image_path, pixmap):
        self.label.setPixmap(pixmap)
