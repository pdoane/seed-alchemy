from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from . import configuration, utils


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Dialog | Qt.FramelessWindowHint)
        self.setWindowTitle("About")

        logo_label = QLabel()
        with Image.open(configuration.get_resource_path("app_icon.png")) as image:
            image.thumbnail((128, 128), Image.Resampling.LANCZOS, reducing_gap=None)
            image_pixmap = QPixmap(utils.pil_to_qimage(image))
        logo_label.setPixmap(image_pixmap)

        app_info_label = QLabel(f"{configuration.APP_NAME}\nVersion {configuration.APP_VERSION}")
        app_info_label.setAlignment(Qt.AlignCenter)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(logo_label)
        layout.addWidget(app_info_label)
        layout.addWidget(ok_button)

        self.setLayout(layout)
