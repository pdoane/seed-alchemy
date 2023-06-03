from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QStyle, QVBoxLayout

from . import utils


class DeleteImageDialog(QDialog):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Confirm Delete")

        # Widgets
        icon_label = QLabel()
        icon_label.setPixmap(self.style().standardPixmap(QStyle.StandardPixmap.SP_MessageBoxWarning))
        icon_label.setAlignment(Qt.AlignTop)

        bold_font = QFont()
        bold_font.setWeight(QFont.Bold)

        message_label = QLabel("Are you sure you want to delete this image?")
        message_label.setFont(bold_font)

        image_label = QLabel()
        with Image.open(image_path) as image:
            image.thumbnail((384, 384), Image.Resampling.LANCZOS, reducing_gap=None)
            image_pixmap = QPixmap(utils.pil_to_qimage(image))
        image_label.setPixmap(image_pixmap)

        yes_button = QPushButton("Yes")
        yes_button.setDefault(True)
        no_button = QPushButton("No")

        # Layout
        message_and_image_layout = QVBoxLayout()
        message_and_image_layout.addWidget(message_label)
        message_and_image_layout.addWidget(image_label)

        icon_and_message_layout = QHBoxLayout()
        icon_and_message_layout.setSpacing(16)
        icon_and_message_layout.addWidget(icon_label)
        icon_and_message_layout.addLayout(message_and_image_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(no_button)
        button_layout.addWidget(yes_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(icon_and_message_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # Slots
        yes_button.clicked.connect(self.accept)
        no_button.clicked.connect(self.reject)
