import json

from PySide6.QtCore import QRectF, QSettings, Qt, QTimer
from PySide6.QtWidgets import QFrame, QGraphicsScene, QHBoxLayout, QListView, QVBoxLayout, QWidget, QButtonGroup

from . import actions
from .backend import Backend
from .canvas_view import CanvasView
from .canvas_generation_area import CanvasGenerationArea
from .canvas_image_element import CanvasImageElement
from .canvas_list_model import CanvasListModel
from .canvas_state import *


class CanvasModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.backend: Backend = main_window.backend
        self.settings: QSettings = main_window.settings
        self.canvas_state = CanvasState()

        self.canvas_scene = QGraphicsScene()
        self.canvas_scene.setSceneRect(-16 * 1024, -16 * 1024, 32 * 1024, 32 * 1024)

        self.generation_area = CanvasGenerationArea(QRectF(0, 0, 768, 768))
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

        mode_layout = QHBoxLayout(self)
        mode_layout.setContentsMargins(8, 2, 8, 8)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(tool_frame)
        mode_layout.addWidget(self.canvas_view)
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

    def add_image(self, image_path):
        pixmap_item = CanvasImageElement(self.canvas_state)
        pixmap_item.set_image(image_path)
        self.canvas_scene.addItem(pixmap_item)
        self.canvas_view.centerOn(pixmap_item)

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

    def get_menus(self):
        return []

    def on_close(self):
        return True

    def on_key_press(self, event):
        return False
