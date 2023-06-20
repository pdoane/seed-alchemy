import json
import os

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import actions, canvas_tool, configuration
from . import font_awesome as fa
from .backend import Backend
from .canvas_generation_element import CanvasGenerationElement
from .canvas_image_element import CanvasImageElement
from .canvas_layer_panel import CanvasLayerPanel
from .canvas_scene import CanvasScene
from .generate_thread import GenerateImageTask
from .image_generation_panel import ImageGenerationPanel
from .pipelines import GenerateRequest
from .image_metadata import ImageMetadata


class CanvasModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.backend: Backend = main_window.backend
        self.settings: QSettings = main_window.settings
        self.generate_task = None
        self.generation_element = None

        self.generation_panel = ImageGenerationPanel(main_window, "canvas")
        self.generation_panel.generate_requested.connect(self.generate_requested)
        self.generation_panel.cancel_requested.connect(self.cancel_requested)
        self.generation_panel.image_size_changed.connect(self.panel_image_size_changed)

        selection_button = actions.selection.tool_button()
        brush_button = actions.brush.tool_button()
        eraser_button = actions.eraser.tool_button()

        self.tool_button_group = QButtonGroup()
        self.tool_button_group.addButton(selection_button, canvas_tool.SELECTION)
        self.tool_button_group.addButton(brush_button, canvas_tool.BRUSH)
        self.tool_button_group.addButton(eraser_button, canvas_tool.ERASER)
        self.tool_button_group.idToggled.connect(self.on_tool_changed)

        tool_frame = QFrame()
        tool_frame.setFrameStyle(QFrame.Panel)

        tool_layout = QVBoxLayout(tool_frame)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(0)
        tool_layout.addWidget(selection_button)
        tool_layout.addWidget(brush_button)
        tool_layout.addWidget(eraser_button)
        tool_layout.addStretch()

        self.canvas_scene = CanvasScene()
        self.layer_panel = CanvasLayerPanel(self.canvas_scene)

        prev_image_button = QPushButton("<")
        prev_image_button.clicked.connect(self.on_prev_image)
        next_image_button = QPushButton(">")
        next_image_button.clicked.connect(self.on_next_image)
        delete_image_button = QPushButton(fa.icon_trash)
        delete_image_button.setFont(fa.font)
        delete_image_button.clicked.connect(self.on_delete_image)
        accept_image_button = QPushButton("Accept")
        accept_image_button.clicked.connect(self.on_accept_image)
        flatten_image_button = QPushButton("Flatten")
        flatten_image_button.clicked.connect(self.on_flatten_image)

        button_layout = QHBoxLayout()
        button_layout.addWidget(prev_image_button)
        button_layout.addWidget(next_image_button)
        button_layout.addWidget(delete_image_button)
        button_layout.addWidget(accept_image_button)
        button_layout.addWidget(flatten_image_button)

        scene_vlayout = QVBoxLayout()
        scene_vlayout.setContentsMargins(0, 0, 0, 0)
        scene_vlayout.addLayout(button_layout)
        scene_vlayout.addWidget(self.canvas_scene)

        mode_layout = QHBoxLayout(self)
        mode_layout.setContentsMargins(8, 2, 8, 8)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(self.generation_panel)
        mode_layout.addWidget(tool_frame)
        mode_layout.addLayout(scene_vlayout)
        mode_layout.addWidget(self.layer_panel)

        # Deserialize scene
        self.canvas_scene.deserialize(json.loads(self.settings.value("canvas", "{}")))

        for element in self.canvas_scene.elements():
            # TODO - multiple generators
            if type(element) == CanvasGenerationElement:
                self.generation_element = element

        if self.generation_element is None:
            self.generation_element = CanvasGenerationElement(self.canvas_scene)
            self.generation_element.set_size(self.generation_panel.get_image_size())
            self.canvas_scene.add_element(self.generation_element)

        self.generation_element.image_size_changed.connect(self.generation_image_size_changed)

        self.tool_button_group.button(self.canvas_scene.tool).setChecked(True)

        # Deserialize panel
        self.generation_panel.deserialize(self.generation_element.params())

        # Serialization
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.serialize)

    def showEvent(self, event):
        self.timer.start(1000)

    def hideEvent(self, event):
        self.timer.stop()

    def get_menus(self):
        return []

    def on_close(self):
        if self.generate_task:
            self.generate_task.cancel = True
        return True

    def on_key_press(self, event):
        return False

    def add_image(self, image_path):
        element = CanvasImageElement(self.canvas_scene)
        element.set_pos(self.generation_element.rect().topLeft())
        element.set_image(image_path)
        self.canvas_scene.add_element(element)

    def serialize(self):
        self.settings.setValue("canvas", json.dumps(self.canvas_scene.serialize()))

    def on_tool_changed(self, button_id, checked):
        if not checked:
            return

        self.canvas_scene.tool = button_id

    def on_prev_image(self):
        self.generation_element.prev_image()

    def on_next_image(self):
        self.generation_element.next_image()

    def on_delete_image(self):
        self.generation_element.delete_image()

    def on_accept_image(self):
        self.generation_element.accept_image()

    def on_flatten_image(self):
        self.generation_element.flatten_image()

    def generate_requested(self):
        if self.generate_task:
            return

        params = self.generation_panel.serialize()
        self.generation_element.set_params(params)
        self.serialize()

        self.generation_element.inpaint_image()

        req = GenerateRequest()
        req.collection = configuration.CANVAS_DIR
        req.reduce_memory = self.settings.value("reduce_memory", type=bool)
        req.image_metadata = ImageMetadata()
        req.image_metadata.load_from_params(params)
        req.num_images_per_prompt = params["num_images_per_prompt"]

        self.generation_panel.begin_generate()
        self.generate_task = GenerateImageTask(req)
        self.generate_task.task_progress.connect(self.update_progress)
        self.generate_task.image_preview.connect(self.image_preview)
        self.generate_task.image_complete.connect(self.image_complete)
        self.generate_task.completed.connect(self.generate_complete)
        self.backend.start(self.generate_task)

    def cancel_requested(self):
        if self.generate_task:
            self.generate_task.cancel = True

    def update_progress(self, progress_amount):
        self.backend.update_progress(progress_amount)

    def image_preview(self, preview_image: Image.Image):
        self.generation_element.set_preview_image(preview_image)

    def image_complete(self, output_path):
        self.generation_element.add_image(output_path)

    def generate_complete(self):
        self.generation_panel.end_generate()
        self.generation_element.set_preview_image(None)
        self.generate_task = None

    def panel_image_size_changed(self, image_size):
        self.generation_element.set_size(image_size)

    def generation_image_size_changed(self, image_size):
        self.generation_panel.set_image_size(image_size)
