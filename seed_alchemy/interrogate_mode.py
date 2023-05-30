import os
import random

from clip_interrogator import Config, Interrogator
from clip_interrogator.clip_interrogator import CAPTION_MODELS
from PIL import Image
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import configuration
from .prompt_result_widget import PromptResultWidget
from .prompt_text_edit import PromptTextEdit
from .source_image_widget import SourceImageWidget
from .thumbnail_loader import ThumbnailLoader
from .thumbnail_model import ThumbnailModel
from .widgets import ComboBox, IntSliderSpinBox, ScrollArea


class InterrogateModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.settings: QSettings = main_window.settings
        self.thumbnail_loader = ThumbnailLoader()
        QApplication.instance().aboutToQuit.connect(self.thumbnail_loader.shutdown)
        self.thumbnail_model = ThumbnailModel(self.thumbnail_loader, 100)

        # Interrogate
        self.interrogate_button = QPushButton("Interrogate")
        self.interrogate_button.clicked.connect(self.on_interrogate)

        # Image
        self.source_image_widget = SourceImageWidget(self.thumbnail_loader, self.thumbnail_model)

        image_group_box = QGroupBox("Image")
        image_group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        image_layout = QVBoxLayout(image_group_box)
        image_layout.addWidget(self.source_image_widget)

        # General
        caption_model_label = QLabel("Caption Model")
        caption_model_label.setAlignment(Qt.AlignCenter)
        self.caption_model_combo_box = ComboBox()
        self.caption_model_combo_box.addItems(CAPTION_MODELS.keys())
        caption_model_layout = QVBoxLayout()
        caption_model_layout.setContentsMargins(0, 0, 0, 0)
        caption_model_layout.setSpacing(2)
        caption_model_layout.addWidget(caption_model_label)
        caption_model_layout.addWidget(self.caption_model_combo_box)

        clip_model_label = QLabel("Clip Model")
        clip_model_label.setAlignment(Qt.AlignCenter)
        self.clip_model_combo_box = ComboBox()
        self.clip_model_combo_box.addItems(["ViT-L-14/openai"])
        clip_model_layout = QVBoxLayout()
        clip_model_layout.setContentsMargins(0, 0, 0, 0)
        clip_model_layout.setSpacing(2)
        clip_model_layout.addWidget(clip_model_label)
        clip_model_layout.addWidget(self.clip_model_combo_box)

        mode_label = QLabel("Mode")
        mode_label.setAlignment(Qt.AlignCenter)
        self.mode_combo_box = ComboBox()
        self.mode_combo_box.addItems(["Best", "Classic", "Fast", "Negative"])
        mode_layout = QVBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(2)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo_box)

        self.max_caption_length = IntSliderSpinBox("Max Caption Length", minimum=1, maximum=75)
        self.min_flavors = IntSliderSpinBox("Min Flavors", minimum=1, maximum=64)
        self.max_flavors = IntSliderSpinBox("Max Flavors", minimum=1, maximum=64)

        general_group_box = QGroupBox("General")
        general_group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        general_layout = QVBoxLayout(general_group_box)
        general_layout.addLayout(caption_model_layout)
        general_layout.addLayout(clip_model_layout)
        general_layout.addLayout(mode_layout)
        general_layout.addWidget(self.max_caption_length)
        general_layout.addWidget(self.min_flavors)
        general_layout.addWidget(self.max_flavors)

        # Manual Caption
        self.caption_prompt = PromptTextEdit(8, "Image Caption")

        self.manual_caption_group_box = QGroupBox("Manual Caption")
        self.manual_caption_group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.manual_caption_group_box.setCheckable(True)
        manual_caption_layout = QVBoxLayout(self.manual_caption_group_box)
        manual_caption_layout.addWidget(self.caption_prompt)

        # Configuration
        self.config_frame = QFrame()
        self.config_frame.setContentsMargins(0, 0, 2, 0)

        self.config_scroll_area = ScrollArea()
        self.config_scroll_area.setFrameStyle(QFrame.NoFrame)
        self.config_scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.config_scroll_area.setWidgetResizable(True)
        self.config_scroll_area.setWidget(self.config_frame)
        self.config_scroll_area.setFocusPolicy(Qt.NoFocus)

        config_layout = QVBoxLayout(self.config_frame)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(image_group_box)
        config_layout.addWidget(general_group_box)
        config_layout.addWidget(self.manual_caption_group_box)
        config_layout.addStretch()

        # Panel
        panel_layout = QVBoxLayout()
        panel_layout.addWidget(self.interrogate_button)
        panel_layout.addWidget(self.config_scroll_area)

        # Results
        self.result_widget = PromptResultWidget(self.main_window)
        self.result_widget.hide()

        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.Panel)

        result_layout = QVBoxLayout(result_frame)
        result_layout.addWidget(self.result_widget)
        result_layout.addStretch()

        # Top-level layout
        prompt_mode_layout = QHBoxLayout(self)
        prompt_mode_layout.setContentsMargins(8, 2, 8, 8)
        prompt_mode_layout.setSpacing(8)
        prompt_mode_layout.addLayout(panel_layout)
        prompt_mode_layout.addWidget(result_frame)

        # Set current values
        self.settings.beginGroup("interrogate")
        self.source_image_widget.line_edit.setText(self.settings.value("source", type=str))
        self.caption_model_combo_box.setCurrentText(self.settings.value("caption_model", type=str))
        self.clip_model_combo_box.setCurrentText(self.settings.value("clip_model", type=str))
        self.mode_combo_box.setCurrentText(self.settings.value("mode", type=str))
        self.max_caption_length.setValue(self.settings.value("max_caption_length", type=int))
        self.min_flavors.setValue(self.settings.value("min_flavors", type=int))
        self.max_flavors.setValue(self.settings.value("max_flavors", type=int))

        self.manual_caption_group_box.setChecked(self.settings.value("manual_caption", type=bool))
        self.caption_prompt.setPlainText(self.settings.value("caption", type=str))
        self.settings.endGroup()

    def get_menus(self):
        return []

    def on_close(self):
        return True

    def on_key_press(self, event):
        return False

    def on_interrogate(self):
        image_path = self.source_image_widget.line_edit.text()
        caption_model_name = self.caption_model_combo_box.currentText()
        clip_model_name = self.clip_model_combo_box.currentText()
        mode = self.mode_combo_box.currentText()
        max_caption_length = self.max_caption_length.value()
        min_flavors = self.min_flavors.value()
        max_flavors = self.max_flavors.value()

        ci = Interrogator(
            Config(
                caption_max_length=max_caption_length,
                caption_model_name=caption_model_name,
                clip_model_name=clip_model_name,
                cache_path=os.path.join(configuration.MODELS_PATH, "interrogator"),
                device=configuration.torch_device,
            )
        )

        full_path = os.path.join(configuration.IMAGES_PATH, image_path)
        with Image.open(full_path) as image:
            if not self.manual_caption_group_box.isChecked():
                self.caption_prompt.setPlainText(ci.generate_caption(image))

            caption = self.caption_prompt.toPlainText()

            self.settings.beginGroup("interrogate")
            self.settings.setValue("source", image_path)
            self.settings.setValue("caption_model", caption_model_name)
            self.settings.setValue("clip_model", clip_model_name)
            self.settings.setValue("mode", mode)
            self.settings.setValue("max_caption_length", max_caption_length)
            self.settings.setValue("min_flavors", min_flavors)
            self.settings.setValue("max_flavors", max_flavors)
            self.settings.setValue("manual_caption", self.manual_caption_group_box.isChecked())
            self.settings.setValue("caption", caption)
            self.settings.endGroup()

            if mode == "Best":
                prompt = ci.interrogate(image, min_flavors=min_flavors, max_flavors=max_flavors, caption=caption)
            elif mode == "Classic":
                prompt = ci.interrogate_classic(image, max_flavors=max_flavors, caption=caption)
            elif mode == "Fast":
                prompt = ci.interrogate_fast(image, max_flavors=max_flavors, caption=caption)
            elif mode == "Negative":
                prompt = ci.interrogate_negative(image, max_flavors=max_flavors)

        self.result_widget.show()
        self.result_widget.label.setText(prompt)
