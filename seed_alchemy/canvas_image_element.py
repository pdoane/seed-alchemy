import os

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QCursor, QPainter, QPen, QPixmap

from . import canvas_tool, configuration, utils
from .canvas_element import CanvasElement, register_class
from .canvas_event import CanvasMouseEvent
from .canvas_scene import CanvasScene
from .image_metadata import ImageMetadata


class CanvasImageElement(CanvasElement):
    pixmap_changed = Signal(QPixmap)

    def __init__(self, scene: CanvasScene, parent=None):
        super().__init__(scene, parent)
        self._pos = QPointF()
        self._metadata: ImageMetadata = None
        self._base_pixmap: QPixmap = None
        self._pixmap: QPixmap = None
        self._mask: QPixmap = None
        self._cursor_pos = None
        self._start_scene_pos = None
        self._start_element_pos = None
        self._last_local_pos = None
        self._painting = False

    def serialize(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "pos": (self.pos().x(), self.pos().y()),
            "image": self._metadata.path,
        }

    def deserialize(self, data: dict) -> None:
        self.set_pos(QPointF(*data.get("pos", (0, 0))))
        self.set_image(data.get("image", ""))

    def bounding_rect(self) -> QRectF:
        return self.rect().adjusted(-1, -1, 1, 1)

    def contains_point(self, point: QPointF) -> bool:
        return self.rect().contains(point)

    def draw_content(self, painter: QPainter) -> None:
        painter.drawPixmap(self._pos, self._pixmap)

        if self.selected() or self.hovered():
            painter.save()
            if self.selected():
                painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            else:
                painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
            painter.drawRect(self.rect())
            painter.restore()

        if self._cursor_pos is not None:
            painter.save()
            painter.setCompositionMode(QPainter.CompositionMode_Difference)
            painter.setPen(QPen(Qt.white, 2, Qt.DashLine))
            brush_size = self._scene.brush_size
            painter.drawEllipse(
                QRectF(
                    self._cursor_pos.x() - brush_size / 2,
                    self._cursor_pos.y() - brush_size / 2,
                    brush_size,
                    brush_size,
                )
            )
            painter.restore()

    def mouse_press_event(self, event: CanvasMouseEvent) -> bool:
        self._start_scene_pos = event.scene_pos
        self._start_element_pos = self._pos
        self._last_local_pos = event.scene_pos - self._pos

        if canvas_tool.is_paint_tool(self._scene.tool):
            self._painting = True

        return True

    def mouse_move_event(self, event: CanvasMouseEvent) -> None:
        if self._painting:
            local_pos = event.scene_pos - self._pos
            with QPainter(self._mask) as painter:
                if self._scene.tool == canvas_tool.BRUSH:
                    color = Qt.white
                else:
                    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                    color = Qt.transparent
                painter.setPen(QPen(color, self._scene.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self._last_local_pos, local_pos)

            self._update_pixmap()
            self._last_local_pos = local_pos
            self._cursor_pos = event.scene_pos
        else:
            new_pos = QPointF(self._start_element_pos)

            dx = event.scene_pos.x() - self._start_scene_pos.x()
            dy = event.scene_pos.y() - self._start_scene_pos.y()

            dx = round(dx / 8) * 8
            dy = round(dy / 8) * 8

            new_pos += QPointF(dx, dy)

            self.update()
            self._pos = new_pos
            self.update()

    def mouse_release_event(self, event: CanvasMouseEvent) -> None:
        self._start_scene_pos = None
        self._last_local_pos = None
        self._painting = False

    def hover_move_event(self, event: CanvasMouseEvent) -> QCursor:
        if canvas_tool.is_paint_tool(self._scene.tool):
            self._cursor_pos = event.scene_pos
            self._update_cursor()
            return Qt.BlankCursor
        else:
            return Qt.ArrowCursor

    def hover_enter_event(self) -> None:
        pass

    def hover_leave_event(self) -> None:
        self._cursor_pos = None
        self._update_cursor()

    def pos(self):
        return self._pos

    def set_pos(self, pos: QPointF):
        self._pos = pos

    def rect(self):
        return QRectF(self._pos, self._pixmap.size())

    def metadata(self):
        return self._metadata

    def pixmap(self):
        return self._pixmap

    def get_image(self) -> Image.Image:
        qcolor = self._base_pixmap.toImage()
        qmask = self._mask.toImage()

        W = qcolor.width()
        H = qcolor.height()

        rgb = np.frombuffer(qcolor.constBits(), np.uint8).reshape((H, W, 4))[..., :3][..., ::-1]
        alpha = np.frombuffer(qmask.constBits(), np.uint8).reshape((H, W, 4))[..., 3:]
        rgba = np.concatenate((rgb, alpha), axis=2)
        return Image.fromarray(rgba)

    def set_image(self, image_path: str):
        full_path = os.path.join(configuration.IMAGES_PATH, image_path)
        try:
            with Image.open(full_path) as image:
                self._metadata = ImageMetadata()
                self._metadata.path = image_path
                self._metadata.load_from_image(image)

                # TODO - upscaled data
                image = image.resize((self._metadata.width, self._metadata.height), Image.Resampling.LANCZOS)

                pixmap = QPixmap.fromImage(utils.pil_to_qimage(image))
        except (IOError, OSError):
            self._metadata = ImageMetadata()
            self._metadata.path = ""
            pixmap = QPixmap(512, 512)

        self._base_pixmap = pixmap
        self._mask = QPixmap(pixmap.size())
        self._mask.fill(Qt.white)
        self._pixmap = QPixmap(self._base_pixmap)
        self.pixmap_changed.emit(self._pixmap)

    def _update_pixmap(self):
        self._pixmap.fill(Qt.transparent)
        with QPainter(self._pixmap) as painter:
            painter.drawPixmap(0, 0, self._base_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.drawPixmap(0, 0, self._mask)
        self.pixmap_changed.emit(self._pixmap)
        self.update()

    def _update_cursor(self):
        x = self._scene.brush_size / 2
        rect = self.bounding_rect().adjusted(-x, -x, x, x)
        self._scene.update_scene_rect(rect)


register_class(CanvasImageElement)
