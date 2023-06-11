import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import transformers
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
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
from .backend import Backend, BackendTask
from .prompt_result_widget import PromptResultWidget
from .prompt_text_edit import PromptTextEdit
from .widgets import ComboBox, FloatSliderSpinBox, IntSliderSpinBox, ScrollArea


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class PromptMetadata:
    model: str = "AUTOMATIC/promptgen-lexart"
    prompt: str = ""
    temperature: float = 1.0
    top_k: int = 12
    top_p: float = 1.0
    num_beams: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    min_length: int = 20
    max_length: int = 150
    count: int = 5
    seed: int = 1

    def load_from_settings(self, settings: QSettings):
        settings.beginGroup("promptgen")
        self.model = settings.value("model", type=str)
        self.prompt = settings.value("prompt", type=str)
        self.temperature = settings.value("temperature", type=float)
        self.top_k = settings.value("top_k", type=int)
        self.top_p = settings.value("top_p", type=float)
        self.num_beams = settings.value("num_beams", type=int)
        self.repetition_penalty = settings.value("repetition_penalty", type=float)
        self.length_penalty = settings.value("length_penalty", type=float)
        self.min_length = settings.value("min_length", type=int)
        self.max_length = settings.value("max_length", type=int)
        self.count = settings.value("count", type=int)
        self.seed = settings.value("seed", type=int)
        settings.endGroup()


class GeneratePromptTask(BackendTask):
    results = Signal(list)

    def __init__(self, metadata: PromptMetadata):
        super().__init__()
        self.metadata = metadata

    def run_(self):
        # Model
        repo_id = configuration.promptgen_models[self.metadata.model]

        tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(repo_id)
        # model.to(configuration.torch_device)

        # Input ids
        input_ids = tokenizer(self.metadata.prompt, return_tensors="pt").input_ids
        if input_ids.shape[1] == 0:
            input_ids = torch.asarray([[tokenizer.bos_token_id]], dtype=torch.long)
        input_ids = input_ids.repeat((self.metadata.count, 1))
        # input_ids = input_ids.to(configuration.torch_device)

        # Generate
        set_seed(self.metadata.seed)
        outputs = model.generate(
            input_ids,
            min_length=self.metadata.min_length,
            max_length=self.metadata.max_length,
            do_sample=True,
            num_beams=self.metadata.num_beams,
            temperature=self.metadata.temperature,
            top_k=self.metadata.top_k,
            top_p=self.metadata.top_p,
            repetition_penalty=self.metadata.repetition_penalty,
            length_penalty=self.metadata.length_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        prompts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.results.emit(prompts)


class PromptModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.backend: Backend = main_window.backend
        self.settings: QSettings = main_window.settings
        self.result_widgets: list[PromptResultWidget] = []
        self.generate_task: GeneratePromptTask = None

        # Generate
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.on_generate)

        # Prompt
        self.prompt_edit = PromptTextEdit(8, "Prompt")
        self.prompt_edit.return_pressed.connect(self.on_generate)

        # General
        model_label = QLabel("Model")
        model_label.setAlignment(Qt.AlignCenter)

        self.model_combo_box = ComboBox()
        self.model_combo_box.addItems(configuration.promptgen_models.keys())

        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(2)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo_box)

        self.prompt_count = IntSliderSpinBox("Prompt Count", minimum=1, maximum=100)
        self.temperature = FloatSliderSpinBox("Temperature", minimum=1e-6, maximum=4)
        self.top_k = IntSliderSpinBox("Top K", minimum=1, maximum=50)
        self.top_p = FloatSliderSpinBox("Top P", minimum=0.0, maximum=1.0)
        self.num_beams = IntSliderSpinBox("Beam Count", minimum=1, maximum=8)
        self.repetition_penalty = FloatSliderSpinBox("Repetition Penalty", minimum=1, maximum=4)
        self.length_penalty = FloatSliderSpinBox("Length Penalty", minimum=-10, maximum=10, step=0.1, decimals=1)
        self.min_length = IntSliderSpinBox("Min Length", minimum=1, maximum=75)
        self.max_length = IntSliderSpinBox("Max Length", minimum=1, maximum=75)

        general_group_box = QGroupBox("General")
        general_layout = QVBoxLayout(general_group_box)
        general_layout.addLayout(model_layout)
        general_layout.addWidget(self.prompt_count)
        general_layout.addWidget(self.temperature)
        general_layout.addWidget(self.top_k)
        general_layout.addWidget(self.top_p)
        general_layout.addWidget(self.num_beams)
        general_layout.addWidget(self.repetition_penalty)
        general_layout.addWidget(self.length_penalty)
        general_layout.addWidget(self.min_length)
        general_layout.addWidget(self.max_length)

        # Seed
        self.seed_line_edit = QLineEdit()
        self.seed_line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        seed_random_button = QPushButton("New")
        seed_random_button.clicked.connect(self.on_randomize_seed)

        seed_hlayout = QHBoxLayout()
        seed_hlayout.setContentsMargins(0, 0, 0, 0)
        seed_hlayout.addWidget(self.seed_line_edit)
        seed_hlayout.addWidget(seed_random_button)

        self.manual_seed_group_box = QGroupBox("Manual Seed")
        self.manual_seed_group_box.setCheckable(True)
        manual_seed_layout = QVBoxLayout(self.manual_seed_group_box)
        manual_seed_layout.addLayout(seed_hlayout)

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
        config_layout.addWidget(self.prompt_edit)
        config_layout.addWidget(general_group_box)
        config_layout.addWidget(self.manual_seed_group_box)
        config_layout.addStretch()

        # Panel
        panel_layout = QVBoxLayout()
        panel_layout.addWidget(self.generate_button)
        panel_layout.addWidget(self.config_scroll_area)

        # Results
        self.result_frame = QFrame()
        self.result_frame.setFrameShape(QFrame.Panel)

        self.result_scroll_area = ScrollArea()
        self.result_scroll_area.setFrameStyle(QFrame.NoFrame)
        self.result_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_scroll_area.setWidgetResizable(True)
        self.result_scroll_area.setWidget(self.result_frame)
        self.result_scroll_area.setFocusPolicy(Qt.NoFocus)

        self.result_layout = QVBoxLayout(self.result_frame)
        self.result_layout_index = self.result_layout.count()
        self.result_layout.addStretch()

        # Top-level layout
        prompt_mode_layout = QHBoxLayout(self)
        prompt_mode_layout.setContentsMargins(8, 2, 8, 8)
        prompt_mode_layout.setSpacing(8)
        prompt_mode_layout.addLayout(panel_layout)
        prompt_mode_layout.addWidget(self.result_scroll_area)

        # Set current values
        self.settings.beginGroup("promptgen")
        self.prompt_edit.setPlainText(self.settings.value("prompt", type=str))
        self.prompt_count.setValue(self.settings.value("count", type=int))
        self.model_combo_box.setCurrentText(self.settings.value("model", type=str))
        self.temperature.setValue(self.settings.value("temperature", type=float))
        self.top_k.setValue(self.settings.value("top_k", type=int))
        self.top_p.setValue(self.settings.value("top_p", type=float))
        self.num_beams.setValue(self.settings.value("num_beams", type=int))
        self.repetition_penalty.setValue(self.settings.value("repetition_penalty", type=float))
        self.length_penalty.setValue(self.settings.value("length_penalty", type=float))
        self.min_length.setValue(self.settings.value("min_length", type=int))
        self.max_length.setValue(self.settings.value("max_length", type=int))

        self.manual_seed_group_box.setChecked(self.settings.value("manual_seed", type=bool))
        self.seed_line_edit.setText(self.settings.value("seed", type=str))
        self.settings.endGroup()

    def get_menus(self):
        return []

    def on_close(self):
        return True

    def on_key_press(self, event):
        return False

    def on_randomize_seed(self):
        self.randomize_seed()

    def randomize_seed(self):
        seed = random.randint(0, 0x7FFF_FFFF)
        self.seed_line_edit.setText(str(seed))

    def on_generate(self):
        if self.generate_task is not None:
            return
        if not self.manual_seed_group_box.isChecked():
            self.randomize_seed()

        self.settings.beginGroup("promptgen")
        self.settings.setValue("prompt", self.prompt_edit.toPlainText())
        self.settings.setValue("model", self.model_combo_box.currentText())
        self.settings.setValue("temperature", self.temperature.value())
        self.settings.setValue("top_k", self.top_k.value())
        self.settings.setValue("top_p", self.top_p.value())
        self.settings.setValue("num_beams", self.num_beams.value())
        self.settings.setValue("repetition_penalty", self.repetition_penalty.value())
        self.settings.setValue("length_penalty", self.length_penalty.value())
        self.settings.setValue("min_length", self.min_length.value())
        self.settings.setValue("max_length", self.max_length.value())
        self.settings.setValue("count", self.prompt_count.value())
        self.settings.setValue("seed", int(self.seed_line_edit.text()))
        self.settings.setValue("manual_seed", self.manual_seed_group_box.isChecked())
        self.settings.endGroup()

        self.metadata = PromptMetadata()
        self.metadata.load_from_settings(self.settings)

        self.generate_button.setEnabled(False)
        self.generate_task = GeneratePromptTask(self.metadata)
        self.generate_task.results.connect(self.on_results)
        self.backend.start(self.generate_task)

    def on_results(self, prompts):
        while len(self.result_widgets) < self.metadata.count:
            result_widget = PromptResultWidget(self.main_window)
            self.result_layout.insertWidget(self.result_layout_index, result_widget)
            self.result_widgets.append(result_widget)

        while len(self.result_widgets) > self.metadata.count:
            result_widget = self.result_widgets.pop()
            self.result_layout.removeWidget(result_widget)
            result_widget.setParent(None)

        for i, prompt in enumerate(prompts):
            result_widget = self.result_widgets[i]
            result_widget.set_text(prompt)

        self.generate_button.setEnabled(True)
        self.generate_task = None
