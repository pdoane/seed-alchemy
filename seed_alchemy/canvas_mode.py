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

        self.generation_panel = ImageGenerationPanel(main_window)
        self.generation_panel.generate_requested.connect(self.generate_requested)
        self.generation_panel.cancel_requested.connect(self.cancel_requested)
        self.generation_panel.image_size_changed.connect(self.panel_image_size_changed)

        self.canvas_scene = CanvasScene()

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

        self.layer_panel = CanvasLayerPanel(self.canvas_scene)

        composite_button = QPushButton("Composite")
        composite_button.clicked.connect(self.on_composite_clicked)

        vlayout = QVBoxLayout()
        vlayout.addWidget(composite_button)
        vlayout.addWidget(self.canvas_scene)

        mode_layout = QHBoxLayout(self)
        mode_layout.setContentsMargins(8, 2, 8, 8)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(self.generation_panel)
        mode_layout.addWidget(tool_frame)
        mode_layout.addLayout(vlayout)
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

    def generate_requested(self):
        if self.generate_task:
            return

        params = self.generation_panel.serialize()
        self.generation_element.set_params(params)

        req = GenerateRequest()
        req.collection = self.settings.value("collection")
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

    def image_preview(self, preview_image):
        pass

    def image_complete(self, output_path):
        self.generation_element.add_image(output_path)

    def generate_complete(self):
        self.generation_panel.end_generate()
        self.generate_task = None

    def panel_image_size_changed(self, image_size):
        self.generation_element.set_size(image_size)

    def generation_image_size_changed(self, image_size):
        self.generation_panel.set_image_size(image_size)

    def on_composite_clicked(self):
        rect = self.generation_element.rect()

        origin = rect.topLeft().toPoint()
        composite_size = (int(rect.width()), int(rect.height()))
        composite_image = Image.new("RGBA", composite_size)

        for element in self.canvas_scene.elements():
            if type(element) == CanvasImageElement:
                image_element: CanvasImageElement = element
                pimage = image_element.get_image()

                pos = element.pos().toPoint()
                composite_image.paste(pimage, (pos - origin).toTuple())

        output_path = "composite.png"
        full_path = os.path.join(configuration.IMAGES_PATH, output_path)
        composite_image.save(full_path)
