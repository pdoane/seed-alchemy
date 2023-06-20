from typing import Optional

from PySide6.QtCore import QObject, QPointF, QRectF, Signal
from PySide6.QtGui import QCursor, QPainter

from .canvas_event import CanvasMouseEvent


class CanvasElement(QObject):
    hovered_changed = Signal()
    selection_changed = Signal()

    def __init__(self, scene, parent) -> None:
        super().__init__(parent)
        self._scene = scene
        self._hovered = False
        self._selected = False

    def scene(self):
        return self._scene

    def hovered(self):
        return self._hovered

    def selected(self):
        return self._selected

    def set_hovered(self, hovered):
        if self._hovered != hovered:
            self._hovered = hovered
            self.hovered_changed.emit()
            self.update()

    def set_selected(self, selected):
        if self._selected != selected:
            self._selected = selected
            self.selection_changed.emit()
            self.update()

    def update(self):
        self._scene.update_element(self)

    def serialize(self) -> dict:
        return {}

    def deserialize(self, data: dict) -> None:
        pass

    def layer(self):
        return 1

    def bounding_rect(self) -> QRectF:
        return QRectF()

    def contains_point(self, point: QPointF) -> bool:
        return self.bounding_rect().contains(point)

    def draw_background(self, painter: QPainter) -> None:
        pass

    def draw_content(self, painter: QPainter) -> None:
        pass

    def draw_controls(self, painter: QPainter) -> None:
        pass

    def mouse_press_event(self, event: CanvasMouseEvent) -> bool:
        return False

    def mouse_move_event(self, event: CanvasMouseEvent) -> None:
        pass

    def mouse_release_event(self, event: CanvasMouseEvent) -> None:
        pass

    def accepts_hover(self, event: CanvasMouseEvent) -> bool:
        return True

    def hover_move_event(self, event: CanvasMouseEvent) -> Optional[QCursor]:
        return None

    def hover_enter_event(self) -> None:
        pass

    def hover_leave_event(self) -> None:
        pass


_class_dict = {}


def register_class(cls):
    _class_dict[cls.__name__] = cls


def deserialize_element(cls_name, scene, data):
    cls = _class_dict.get(cls_name)
    if cls is not None:
        element: CanvasElement = cls(scene)
        element.deserialize(data)
        return element
    return None
