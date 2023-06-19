from PySide6.QtCore import QPointF, QRectF, QSize, QSizeF, Qt, Signal
from PySide6.QtGui import QCursor, QPainter, QPen

from . import utils
from .canvas_scene import CanvasScene
from .canvas_element import CanvasElement, register_class
from .canvas_event import CanvasMouseEvent
from .canvas_image_element import CanvasImageElement

HANDLE_SIZE = 5


class CanvasGenerationElement(CanvasElement):
    image_size_changed = Signal(QSize)

    def __init__(self, scene: CanvasScene, parent=None):
        super().__init__(scene, parent)
        self._images: list[CanvasImageElement] = []
        self._rect = QRectF()
        self._params = {}
        self._background_brush = utils.checkerboard_qbrush()
        self._start_scene_pos = None
        self._start_rect = None
        self._resize_x = None
        self._resize_y = None
        self._cursor = None

    def serialize(self):
        return {
            "class": self.__class__.__name__,
            "rect": (self.rect().left(), self.rect().top(), self.rect().width(), self.rect().height()),
            "params": self._params,
            "images": [image.serialize() for image in self._images],
        }

    def deserialize(self, data):
        self.set_rect(QRectF(*data.get("rect", (0, 0, 512, 512))))
        self._params = data.get("params", {})

        for image_data in data.get("images", []):
            image = CanvasImageElement(self._scene)
            image.deserialize(image_data)
            image.set_pos(self._rect.topLeft())
            self._images.append(image)

    def bounding_rect(self) -> QRectF:
        return self._rect.adjusted(-HANDLE_SIZE, -HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE)

    def contains_point(self, point: QPointF) -> bool:
        if self._rect.contains(point):
            return True
        for handle in self._get_handles():
            if handle.contains(point):
                return True
        return False

    def draw_background(self, painter: QPainter) -> None:
        painter.save()
        painter.fillRect(self._rect, self._background_brush)
        painter.restore()

    def draw_content(self, painter: QPainter) -> None:
        for image in self._images:
            image.draw_content(painter)

    def draw_controls(self, painter: QPainter) -> None:
        painter.save()
        if self.selected():
            painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        elif self.hovered():
            painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
        else:
            painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
        painter.drawRect(self._rect)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        painter.drawRects(self._get_handles())
        painter.restore()

    def mouse_press_event(self, event: CanvasMouseEvent) -> bool:
        self._start_scene_pos = event.scene_pos
        self._start_rect = self._rect
        cursor, resize_x, resize_y = self._get_resize_info(event.scene_pos)
        self._resize_x = resize_x
        self._resize_y = resize_y
        self._cursor = cursor
        return True

    def mouse_move_event(self, event: CanvasMouseEvent) -> None:
        if self._start_scene_pos is None:
            return

        new_rect = QRectF(self._start_rect)

        dx = event.scene_pos.x() - self._start_scene_pos.x()
        dy = event.scene_pos.y() - self._start_scene_pos.y()

        dx = round(dx / 8) * 8
        dy = round(dy / 8) * 8

        if self._cursor == Qt.ArrowCursor:
            new_rect.translate(dx, dy)
        else:
            if self._resize_x == "left":
                new_rect.setLeft(self._start_rect.left() + dx)
            elif self._resize_x == "right":
                new_rect.setRight(self._start_rect.right() + dx)

            if self._resize_y == "top":
                new_rect.setTop(self._start_rect.top() + dy)
            elif self._resize_y == "bottom":
                new_rect.setBottom(self._start_rect.bottom() + dy)

        if new_rect.width() < 8:
            new_rect.setWidth(8)
        if new_rect.height() < 8:
            new_rect.setHeight(8)

        self.update()
        self.set_rect(new_rect)
        self.update()

    def mouse_release_event(self, event: CanvasMouseEvent) -> None:
        self._start_scene_pos = None
        self._start_rect = None
        self._cursor = None
        self._resize_x = None
        self._resize_y = None

    def hover_move_event(self, event: CanvasMouseEvent) -> QCursor:
        cursor, _, _ = self._get_resize_info(event.scene_pos)
        return cursor

    def rect(self):
        return self._rect

    def set_rect(self, rect):
        self._rect = rect
        for image in self._images:
            image.set_pos(rect.topLeft())
        self.image_size_changed.emit(rect.size().toSize())

    def set_size(self, size):
        rect = QRectF(self._rect)
        rect.setSize(size)
        self.set_rect(rect)

    def add_image(self, image_path):
        image = CanvasImageElement(self._scene)
        image.set_image(image_path)
        image.set_pos(self._rect.topLeft())
        self._images.append(image)
        self.update()

    def params(self):
        return self._params

    def set_params(self, params):
        self._params = params

    def _get_handles(self):
        offset = QPointF(-HANDLE_SIZE, -HANDLE_SIZE)
        size = QSizeF(HANDLE_SIZE * 2, HANDLE_SIZE * 2)
        return [
            QRectF(self._rect.topLeft() + offset, size),
            QRectF(self._rect.topRight() + offset, size),
            QRectF(self._rect.bottomLeft() + offset, size),
            QRectF(self._rect.bottomRight() + offset, size),
            QRectF(QPointF(self._rect.center().x(), self._rect.top()) + offset, size),
            QRectF(QPointF(self._rect.center().x(), self._rect.bottom()) + offset, size),
            QRectF(QPointF(self._rect.left(), self._rect.center().y()) + offset, size),
            QRectF(QPointF(self._rect.right(), self._rect.center().y()) + offset, size),
        ]

    def _get_resize_info(self, pos):
        resize_info = [
            (Qt.SizeFDiagCursor, "left", "top"),
            (Qt.SizeBDiagCursor, "right", "top"),
            (Qt.SizeBDiagCursor, "left", "bottom"),
            (Qt.SizeFDiagCursor, "right", "bottom"),
            (Qt.SizeVerCursor, None, "top"),
            (Qt.SizeVerCursor, None, "bottom"),
            (Qt.SizeHorCursor, "left", None),
            (Qt.SizeHorCursor, "right", None),
        ]

        for i, rect in enumerate(self._get_handles()):
            if rect.contains(pos):
                return resize_info[i]

        return Qt.ArrowCursor, None, None


register_class(CanvasGenerationElement)
