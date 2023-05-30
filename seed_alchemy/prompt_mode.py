import random

import numpy as np
import torch
import transformers
from PySide6.QtCore import QSettings, Qt
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
from .prompt_text_edit import PromptTextEdit
from .widgets import ComboBox, FloatSliderSpinBox, IntSliderSpinBox, ScrollArea
from .prompt_result_widget import PromptResultWidget


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class PromptModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.settings: QSettings = main_window.settings
        self.result_widgets: list[PromptResultWidget] = []

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
        if not self.manual_seed_group_box.isChecked():
            self.randomize_seed()

        prompt = self.prompt_edit.toPlainText()
        model_name = self.model_combo_box.currentText()
        temperature = self.temperature.value()
        top_k = self.top_k.value()
        top_p = self.top_p.value()
        num_beams = self.num_beams.value()
        repetition_penalty = self.repetition_penalty.value()
        length_penalty = self.length_penalty.value()
        min_length = self.min_length.value()
        max_length = self.max_length.value()
        count = self.prompt_count.value()
        seed = int(self.seed_line_edit.text())

        self.settings.beginGroup("promptgen")
        self.settings.setValue("prompt", prompt)
        self.settings.setValue("model", model_name)
        self.settings.setValue("temperature", temperature)
        self.settings.setValue("top_k", top_k)
        self.settings.setValue("top_p", top_p)
        self.settings.setValue("num_beams", num_beams)
        self.settings.setValue("repetition_penalty", repetition_penalty)
        self.settings.setValue("length_penalty", length_penalty)
        self.settings.setValue("min_length", min_length)
        self.settings.setValue("max_length", max_length)
        self.settings.setValue("count", count)
        self.settings.setValue("seed", seed)
        self.settings.setValue("manual_seed", self.manual_seed_group_box.isChecked())
        self.settings.endGroup()

        while len(self.result_widgets) < count:
            result_widget = PromptResultWidget(self.main_window)
            self.result_layout.insertWidget(self.result_layout_index, result_widget)
            self.result_widgets.append(result_widget)

        while len(self.result_widgets) > count:
            result_widget = self.result_widgets.pop()
            self.result_layout.removeWidget(result_widget)
            result_widget.setParent(None)

        # Model
        repo_id = configuration.promptgen_models[model_name]

        tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(repo_id)
        # model.to(configuration.torch_device)

        # Input ids
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if input_ids.shape[1] == 0:
            input_ids = torch.asarray([[tokenizer.bos_token_id]], dtype=torch.long)
        input_ids = input_ids.repeat((count, 1))
        # input_ids = input_ids.to(configuration.torch_device)

        # Generate
        set_seed(seed)
        outputs = model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            do_sample=True,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        prompts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, prompt in enumerate(prompts):
            result_widget = self.result_widgets[i]
            result_widget.label.setText(prompt)
