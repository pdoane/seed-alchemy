from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from .backend import Backend
from .canvas_state import *


class GalleryModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.backend: Backend = main_window.backend
        self.settings: QSettings = main_window.settings

        label = QLabel("Image Gallery/Slideshow Coming Soon!")
        label.setAlignment(Qt.AlignCenter)

        mode_layout = QHBoxLayout(self)
        mode_layout.setContentsMargins(8, 2, 8, 8)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(label)

    def get_menus(self):
        return []

    def on_close(self):
        return True

    def on_key_press(self, event):
        return False
