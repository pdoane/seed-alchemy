from dataclasses import dataclass
import os

from clip_interrogator import Config, Interrogator
from clip_interrogator.clip_interrogator import CAPTION_MODELS
from PIL import Image
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import configuration
from .backend import Backend, BackendTask
from .prompt_result_widget import PromptResultWidget
from .prompt_text_edit import PromptTextEdit
from .source_image_widget import SourceImageWidget
from .thumbnail_loader import ThumbnailLoader
from .thumbnail_model import ThumbnailModel
from .widgets import ComboBox, IntSliderSpinBox, ScrollArea


@dataclass
class InterrogateMetadata:
    source: str = ""
    caption_model: str = "blip-large"
    clip_model: str = "ViT-L-14/openai"
    mode: str = "Best"
    max_caption_length: int = 32
    min_flavors: int = 8
    max_flavors: int = 32
    caption: str = None

    def load_from_settings(self, settings: QSettings):
        settings.beginGroup("interrogate")
        self.source = settings.value("source", type=str)
        self.caption_model = settings.value("caption_model", type=str)
        self.clip_model = settings.value("clip_model", type=str)
        self.mode = settings.value("mode", type=str)
        self.max_caption_length = settings.value("max_caption_length", type=int)
        self.min_flavors = settings.value("min_flavors", type=int)
        self.max_flavors = settings.value("max_flavors", type=int)
        if settings.value("manual_caption", type=bool):
            self.caption = settings.value("caption", type=str)
        settings.endGroup()


class InterrogateTask(BackendTask):
    results = Signal(str, str)

    def __init__(self, metadata: InterrogateMetadata):
        super().__init__()
        self.metadata = metadata

    def run_(self):
        ci = Interrogator(
            Config(
                caption_max_length=self.metadata.max_caption_length,
                caption_model_name=self.metadata.caption_model,
                clip_model_name=self.metadata.clip_model,
                cache_path=os.path.join(configuration.MODELS_PATH, "interrogator"),
                device=configuration.torch_device,
            )
        )

        full_path = os.path.join(configuration.IMAGES_PATH, self.metadata.source)
        with Image.open(full_path) as image:
            caption = self.metadata.caption or ci.generate_caption(image)

            if self.metadata.mode == "Best":
                prompt = ci.interrogate(
                    image,
                    min_flavors=self.metadata.min_flavors,
                    max_flavors=self.metadata.max_flavors,
                    caption=caption,
                )
            elif self.metadata.mode == "Classic":
                prompt = ci.interrogate_classic(image, max_flavors=self.metadata.max_flavors, caption=caption)
            elif self.metadata.mode == "Fast":
                prompt = ci.interrogate_fast(image, max_flavors=self.metadata.max_flavors, caption=caption)
            elif self.metadata.mode == "Negative":
                prompt = ci.interrogate_negative(image, max_flavors=self.metadata.max_flavors)

        self.results.emit(caption, prompt)


class InterrogateModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.backend: Backend = main_window.backend
        self.settings: QSettings = main_window.settings
        self.thumbnail_loader: ThumbnailLoader = main_window.thumbnail_loader
        self.thumbnail_model: ThumbnailModel = main_window.thumbnail_model

        self.interrogate_task: InterrogateTask = None

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
        self.caption_label = QLabel()
        self.caption_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.caption_label.setCursor(Qt.IBeamCursor)
        self.caption_label.setWordWrap(True)

        self.prompt = PromptResultWidget(self.main_window)
        self.prompt.hide()

        self.result_frame = QFrame()
        self.result_frame.setFrameShape(QFrame.Panel)

        result_layout = QVBoxLayout(self.result_frame)
        result_layout.addWidget(self.caption_label)
        result_layout.addWidget(self.prompt)
        result_layout.addStretch()

        # Top-level layout
        prompt_mode_layout = QHBoxLayout(self)
        prompt_mode_layout.setContentsMargins(8, 2, 8, 8)
        prompt_mode_layout.setSpacing(8)
        prompt_mode_layout.addLayout(panel_layout)
        prompt_mode_layout.addWidget(self.result_frame)

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
        if self.interrogate_task is not None:
            return

        self.settings.beginGroup("interrogate")
        self.settings.setValue("source", self.source_image_widget.line_edit.text())
        self.settings.setValue("caption_model", self.caption_model_combo_box.currentText())
        self.settings.setValue("clip_model", self.clip_model_combo_box.currentText())
        self.settings.setValue("mode", self.mode_combo_box.currentText())
        self.settings.setValue("max_caption_length", self.max_caption_length.value())
        self.settings.setValue("min_flavors", self.min_flavors.value())
        self.settings.setValue("max_flavors", self.max_flavors.value())
        self.settings.setValue("manual_caption", self.manual_caption_group_box.isChecked())
        self.settings.setValue("caption", self.caption_prompt.toPlainText())
        self.settings.endGroup()

        self.metadata = InterrogateMetadata()
        self.metadata.load_from_settings(self.settings)

        self.interrogate_button.setEnabled(False)
        self.interrogate_task = InterrogateTask(self.metadata)
        self.interrogate_task.results.connect(self.on_results)
        self.backend.start(self.interrogate_task)

    def on_results(self, caption: str, prompt: str):
        self.prompt.show()
        self.caption_label.setText("Caption: " + caption)
        self.prompt.set_text(prompt)
        self.interrogate_button.setEnabled(True)
        self.interrogate_task = None
