from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from . import configuration


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Dialog | Qt.FramelessWindowHint)
        self.setWindowTitle("About")

        layout = QVBoxLayout()

        app_info_label = QLabel(f"{configuration.APP_NAME}\nVersion {configuration.APP_VERSION}")
        app_info_label.setAlignment(Qt.AlignCenter)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)

        layout.addWidget(app_info_label)
        layout.addWidget(ok_button)

        self.setLayout(layout)
        
