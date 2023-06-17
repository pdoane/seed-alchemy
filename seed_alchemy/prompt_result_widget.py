from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
)

from . import actions


class PromptResultWidget(QFrame):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.set_as_image_prompt_button: QToolButton = None
        self.label: QLabel = None
        self.init_ui()

    def init_ui(self):
        self.setFrameStyle(QFrame.Box)

        self.set_as_image_prompt_button = actions.set_as_image_prompt.tool_button()
        self.set_as_image_prompt_button.clicked.connect(self.on_set_as_image_prompt)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setCursor(Qt.IBeamCursor)

        layout = QHBoxLayout(self)
        layout.addWidget(self.set_as_image_prompt_button)
        layout.addWidget(self.label)

    def set_text(self, str):
        self.label.setText(str)
        self.label.setEnabled(True)

    def on_set_as_image_prompt(self):
        self.label.setEnabled(False)
        image_mode_widget = self.main_window.set_mode("image")
        image_mode_widget.generation_panel.prompt_edit.setPlainText(self.label.text())
