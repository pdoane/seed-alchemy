import math
import os
from typing import Optional

from PIL import Image, ImageQt
from PySide6.QtCore import QPointF, QRectF, QSize, QSizeF, Qt, Signal, QPoint
from PySide6.QtGui import QCursor, QPainter, QPen, QPixmap

from . import canvas_tool, utils, configuration
from .canvas_element import CanvasElement, register_class
from .canvas_event import CanvasMouseEvent
from .canvas_image_element import CanvasImageElement
from .canvas_scene import CanvasScene

HANDLE_SIZE = 5


class CanvasGenerationElement(CanvasElement):
    pixmap_changed = Signal(QPixmap)
    image_size_changed = Signal(QSize)

    def __init__(self, scene: CanvasScene, parent=None):
        super().__init__(scene, parent)
        self._images: list[CanvasImageElement] = []
        self._image_index = 0
        self._preview_pixmap: Optional[QPixmap] = None
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
            "image_index": self._image_index,
        }

    def deserialize(self, data):
        self.set_rect(QRectF(*data.get("rect", (0, 0, 512, 512))))
        self._params = data.get("params", {})
        self.set_image_index(data.get("image_index", 0))

        for image_data in data.get("images", []):
            image = CanvasImageElement(self._scene)
            image.deserialize(image_data)
            image.set_pos(self._rect.topLeft())
            self._images.append(image)

    def layer(self):
        return 0

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
        if self._images:
            image = self._images[self._image_index]
            image.draw_content(painter)

        if self._preview_pixmap is not None:
            painter.drawPixmap(self._rect.topLeft(), self._preview_pixmap)

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
        if self._scene.tool == canvas_tool.SELECTION:
            self._start_scene_pos = event.scene_pos
            self._start_rect = self._rect
            cursor, resize_x, resize_y = self._get_resize_info(event.scene_pos)
            self._resize_x = resize_x
            self._resize_y = resize_y
            self._cursor = cursor
            return True
        return False

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

        self.set_rect(new_rect)

    def mouse_release_event(self, event: CanvasMouseEvent) -> None:
        self._start_scene_pos = None
        self._start_rect = None
        self._cursor = None
        self._resize_x = None
        self._resize_y = None

    def accepts_hover(self, event: CanvasMouseEvent) -> bool:
        return self._scene.tool == canvas_tool.SELECTION

    def hover_move_event(self, event: CanvasMouseEvent) -> QCursor:
        cursor, _, _ = self._get_resize_info(event.scene_pos)
        return cursor

    def rect(self):
        return self._rect

    def set_rect(self, rect):
        self.update()
        self._rect = rect
        self.update()

        for image in self._images:
            image.set_pos(rect.topLeft())
        self.image_size_changed.emit(rect.size().toSize())

    def set_size(self, size):
        rect = QRectF(self._rect)
        rect.setSize(size)
        self.set_rect(rect)

    def pixmap(self):
        if self._images:
            image = self._images[self._image_index]
            return image.pixmap()
        return None

    def params(self):
        return self._params

    def set_params(self, params):
        self._params = params

    def add_image(self, image_path):
        image = CanvasImageElement(self._scene)
        image.set_image(image_path)
        image.set_pos(self._rect.topLeft())
        self._images.append(image)
        self.set_image_index(len(self._images) - 1)

    def set_preview_image(self, preview_image: Optional[Image.Image]):
        if preview_image is not None:
            self._preview_pixmap = ImageQt.toqpixmap(preview_image).scaled(self._rect.width(), self._rect.height())
        else:
            self._preview_pixmap = None
        self.update()

    def prev_image(self):
        self.set_image_index(self._image_index - 1)

    def next_image(self):
        self.set_image_index(self._image_index + 1)

    def delete_image(self):
        if not self._images:
            return
        self._images.pop(self._image_index)
        self.set_image_index(self._image_index)

    def accept_image(self):
        if not self._images:
            return
        image = self._images[self._image_index]
        self._scene.add_element(image)
        self._images.clear()
        self.set_image_index(-1)

    def set_image_index(self, index):
        if self._images:
            self._image_index = (index) % len(self._images)
        else:
            self._image_index = -1
        self.pixmap_changed.emit(self.pixmap())
        self.update()

    def flatten_image(self):
        if not self._images:
            return

        # Find rectangle arond all images the intersect with the generation element
        rect = QRectF(self._rect)
        intersecting_elements = []
        for element in self._scene.elements():
            if type(element) == CanvasImageElement:
                image_element: CanvasImageElement = element
                if image_element.rect().intersects(self._rect):
                    rect = rect.united(image_element.rect())
                    intersecting_elements.append(image_element)

        # Create flattened image
        image = self._images[self._image_index]
        flattened_image = self._make_composite_image(rect)
        flattened_image.paste(
            image.get_image(),
            (image.pos() - rect.topLeft()).toPoint().toTuple(),
            mask=image.get_image(),
        )

        # Save
        def io_operation():
            collection = configuration.CANVAS_DIR
            next_image_id = utils.next_image_id(os.path.join(configuration.IMAGES_PATH, collection))
            output_path = os.path.join(collection, "{:05d}.png".format(next_image_id))
            full_path = os.path.join(configuration.IMAGES_PATH, output_path)

            flattened_image.save(full_path)
            return output_path

        image_path = utils.retry_on_failure(io_operation)

        # Build new element
        image = CanvasImageElement(self._scene)
        image.set_image(image_path)
        image.set_pos(rect.topLeft())

        self._scene.add_element(image)

        # Remove flattened elements
        self._images.clear()
        self.set_image_index(-1)

        for element in intersecting_elements:
            self._scene.remove_element(element)

    def inpaint_image(self):
        composite_image = self._make_composite_image(self._rect)

        def io_operation():
            output_path = os.path.join(configuration.CANVAS_DIR, configuration.INPAINT_IMAGE_NAME)
            full_path = os.path.join(configuration.IMAGES_PATH, output_path)
            composite_image.save(full_path)

        utils.retry_on_failure(io_operation)

    def _make_composite_image(self, rect: QRectF):
        composite_image = Image.new("RGBA", rect.size().toSize().toTuple())

        for element in reversed(self._scene.elements()):
            if type(element) == CanvasImageElement:
                image_element: CanvasImageElement = element
                if image_element.rect().intersects(rect):
                    composite_image.paste(
                        image_element.get_image(),
                        (element.pos() - rect.topLeft()).toPoint().toTuple(),
                        mask=image_element.get_image(),
                    )

        return composite_image

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
