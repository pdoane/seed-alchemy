import json
import os

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsScene,
    QHBoxLayout,
    QListView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import actions, configuration
from .backend import Backend
from .canvas_generation_area import CanvasGenerationArea
from .canvas_image_element import CanvasImageElement
from .canvas_list_model import CanvasListModel
from .canvas_state import *
from .canvas_view import CanvasView
from .generate_thread import GenerateImageTask
from .image_generation_panel import ImageGenerationPanel


class CanvasModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.backend: Backend = main_window.backend
        self.settings: QSettings = main_window.settings
        self.canvas_state = CanvasState()
        self.generate_task = None

        self.generation_panel = ImageGenerationPanel(main_window)
        self.generation_panel.generate_requested.connect(self.generate_requested)
        self.generation_panel.cancel_requested.connect(self.cancel_requested)
        self.generation_panel.image_size_changed.connect(self.panel_image_size_changed)

        self.canvas_scene = QGraphicsScene()
        self.canvas_scene.setSceneRect(-16 * 1024, -16 * 1024, 32 * 1024, 32 * 1024)

        self.generation_area = CanvasGenerationArea()
        self.generation_area.image_size_changed.connect(self.area_image_size_changed)
        self.generation_area.setZValue(1)
        self.canvas_scene.addItem(self.generation_area)

        self.canvas_view = CanvasView()
        self.canvas_view.setScene(self.canvas_scene)

        selection_button = actions.selection.tool_button()
        brush_button = actions.brush.tool_button()
        eraser_button = actions.eraser.tool_button()

        self.tool_button_group = QButtonGroup()
        self.tool_button_group.addButton(selection_button, SELECTION_TOOL)
        self.tool_button_group.addButton(brush_button, BRUSH_TOOL)
        self.tool_button_group.addButton(eraser_button, ERASER_TOOL)
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

        list_view = QListView()
        model = CanvasListModel(self.canvas_scene)
        list_view.setModel(model)
        list_view.setMinimumWidth(300)

        composite_button = QPushButton("Composite")
        composite_button.clicked.connect(self.on_composite_clicked)

        vlayout = QVBoxLayout()
        vlayout.addWidget(composite_button)
        vlayout.addWidget(self.canvas_view)

        mode_layout = QHBoxLayout(self)
        mode_layout.setContentsMargins(8, 2, 8, 8)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(self.generation_panel)
        mode_layout.addWidget(tool_frame)
        mode_layout.addLayout(vlayout)
        mode_layout.addWidget(list_view)

        self.settings.beginGroup("canvas")
        tool = self.settings.value("tool", SELECTION_TOOL, type=int)
        self.tool_button_group.button(tool).setChecked(True)
        self.settings.endGroup()

        self.deserialized = False

    def showEvent(self, event):
        if self.deserialized:
            return
        self.deserialized = True
        self.deserialize()
        timer = QTimer(self)
        timer.timeout.connect(self.serialize)
        timer.start(1000)

    def get_menus(self):
        return []

    def on_close(self):
        if self.generate_task:
            self.generate_task.cancel = True
        return True

    def on_key_press(self, event):
        return False

    def add_image(self, image_path):
        pixmap_item = CanvasImageElement(self.canvas_state)
        pixmap_item.setPos(self.generation_area.rect().topLeft())
        pixmap_item.set_image(image_path)
        self.canvas_scene.addItem(pixmap_item)
        self.canvas_view.ensureVisible(pixmap_item)

    def serialize(self):
        elements_json = json.dumps(
            [item.serialize() for item in self.canvas_scene.items(Qt.AscendingOrder) if item.zValue() == 0]
        )
        view_json = json.dumps(self.canvas_view.serialize())
        generation_area_json = json.dumps(self.generation_area.serialize())

        self.settings.beginGroup("canvas")
        self.settings.setValue("elements", elements_json)
        self.settings.setValue("generation_area", generation_area_json)
        self.settings.setValue("view", view_json)
        self.settings.endGroup()

    def deserialize(self):
        self.settings.beginGroup("canvas")
        elements_json = self.settings.value("elements", "[]")
        generation_area_json = self.settings.value("generation_area", "{}")
        view_json = self.settings.value("view", "{}")
        self.settings.endGroup()

        for item_data in json.loads(elements_json):
            cls = globals()[item_data["class"]]
            item = cls(self.canvas_state)
            item.deserialize(item_data)
            self.canvas_scene.addItem(item)

        self.canvas_view.deserialize(json.loads(view_json))
        self.generation_area.deserialize(json.loads(generation_area_json))

    def on_tool_changed(self, button_id, checked):
        if not checked:
            return

        self.settings.beginGroup("canvas")
        self.settings.setValue("tool", button_id)
        self.settings.endGroup()

        self.canvas_state.tool = button_id

    def generate_requested(self):
        if self.generate_task:
            return

        self.generation_panel.begin_generate()
        self.generate_task = GenerateImageTask(self.settings)
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
        self.add_image(output_path)

    def generate_complete(self):
        self.generation_panel.end_generate()
        self.generate_task = None

    def panel_image_size_changed(self, image_size):
        rect = self.generation_area.rect()
        rect.setSize(image_size)
        self.generation_area.setRect(rect)

    def area_image_size_changed(self, image_size):
        self.generation_panel.set_image_size(image_size)

    def on_composite_clicked(self):
        rect = self.generation_area.rect()

        origin = rect.topLeft().toPoint()
        composite_size = (int(rect.width()), int(rect.height()))
        composite_image = Image.new("RGBA", composite_size)

        for item in self.canvas_scene.items(rect, Qt.IntersectsItemShape, Qt.AscendingOrder):
            if type(item) == CanvasImageElement:
                qcolor = item.base_pixmap.toImage()
                qmask = item.mask.toImage()

                W = qcolor.width()
                H = qcolor.height()

                rgb = np.frombuffer(qcolor.constBits(), np.uint8).reshape((H, W, 4))[..., :3][..., ::-1]
                alpha = np.frombuffer(qmask.constBits(), np.uint8).reshape((H, W, 4))[..., 3:]
                rgba = np.concatenate((rgb, alpha), axis=2)
                pimage = Image.fromarray(rgba)

                pos = item.pos().toPoint()
                composite_image.paste(pimage, (pos - origin).toTuple())

        output_path = "composite.png"
        full_path = os.path.join(configuration.IMAGES_PATH, output_path)
        composite_image.save(full_path)
