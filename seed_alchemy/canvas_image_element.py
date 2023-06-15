import os

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem

from . import configuration, utils
from .canvas_state import *
from .image_metadata import ImageMetadata


class CanvasImageElement(QGraphicsPixmapItem):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.canvas_state: CanvasState = state
        self.metadata: ImageMetadata = None
        self.base_pixmap: QPixmap = None
        self.mask: QPixmap = None
        self.painting = False
        self.cursor_pos = None
        self.brush_size = 30

        self.setAcceptHoverEvents(True)
        self.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def serialize(self):
        return {
            "class": self.__class__.__name__,
            "pos": (self.pos().x(), self.pos().y()),
            "image": self.metadata.path,
        }

    def deserialize(self, data):
        self.setPos(QPointF(*data["pos"]))
        self.set_image(data["image"])

    def describe(self):
        if self.metadata is None:
            return "unknown"
        else:
            return self.metadata.path

    def set_image(self, image_path):
        full_path = os.path.join(configuration.IMAGES_PATH, image_path)
        with Image.open(full_path) as image:
            self.metadata = ImageMetadata()
            self.metadata.path = image_path
            self.metadata.load_from_image(image)

            # TODO - upscaled data
            image = image.resize((self.metadata.width, self.metadata.height), Image.Resampling.LANCZOS)

            pixmap = QPixmap.fromImage(utils.pil_to_qimage(image))

        self.base_pixmap = pixmap
        self.mask = QPixmap(pixmap.size())
        self.mask.fill(Qt.white)

        self.setPixmap(pixmap)

    def paint(self, painter, option, widget):
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(QPointF(0, 0), self.base_pixmap.size()))
        painter.restore()

        super().paint(painter, option, widget)

        if self.isSelected():
            painter.save()
            painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            painter.drawRect(self.boundingRect())
            painter.restore()

        if self.cursor_pos is not None:
            painter.save()
            painter.setCompositionMode(QPainter.CompositionMode_Difference)
            painter.setPen(QPen(Qt.white, 2, Qt.DashLine))
            painter.drawEllipse(
                QRectF(
                    self.cursor_pos.x() - self.brush_size / 2,
                    self.cursor_pos.y() - self.brush_size / 2,
                    self.brush_size,
                    self.brush_size,
                )
            )
            painter.restore()

    def mousePressEvent(self, event):
        if not is_paint_tool(self.canvas_state.tool):
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton and self.base_pixmap is not None:
            self.last_pos = event.pos()
            self.painting = True

    def mouseMoveEvent(self, event):
        if not is_paint_tool(self.canvas_state.tool):
            super().mouseMoveEvent(event)
            return

        cursor_pos = event.pos()

        if (event.buttons() & Qt.LeftButton) and self.painting:
            with QPainter(self.mask) as painter:
                if self.canvas_state.tool == BRUSH_TOOL:
                    color = Qt.white
                else:
                    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                    color = Qt.transparent
                painter.setPen(QPen(color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.last_pos, cursor_pos)

            self.update_pixmap()
            self.last_pos = cursor_pos

        self.cursor_pos = cursor_pos

    def mouseReleaseEvent(self, event):
        if not is_paint_tool(self.canvas_state.tool):
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and self.painting:
            self.painting = False

    def hoverMoveEvent(self, event):
        if not is_paint_tool(self.canvas_state.tool):
            return

        self.cursor_pos = event.pos()
        self.update()

    def hoverEnterEvent(self, event):
        if not is_paint_tool(self.canvas_state.tool):
            return

        self.setCursor(Qt.BlankCursor)

    def hoverLeaveEvent(self, event):
        if not is_paint_tool(self.canvas_state.tool):
            return

        self.setCursor(Qt.ArrowCursor)
        self.cursor_pos = None
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            grid_size = 8
            new_pos = value

            x = round(new_pos.x() / grid_size) * grid_size
            y = round(new_pos.y() / grid_size) * grid_size

            return super().itemChange(change, QPointF(x, y))

        return super().itemChange(change, value)

    def update_pixmap(self):
        new_pixmap = QPixmap(self.base_pixmap.size())
        new_pixmap.fill(Qt.transparent)
        with QPainter(new_pixmap) as painter:
            painter.drawPixmap(0, 0, self.base_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.drawPixmap(0, 0, self.mask)

        self.setPixmap(new_pixmap)
