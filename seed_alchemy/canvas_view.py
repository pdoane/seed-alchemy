import math

from PySide6.QtCore import QLine, QPoint, QRectF, Qt
from PySide6.QtGui import QPainter, QPen, QTransform
from PySide6.QtWidgets import QGraphicsView


class CanvasView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._start_pos = None
        self._selection_rect = None
        self.target_viewport_pos = None
        self.target_scene_pos = None

        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, False)

    def serialize(self):
        transform = self.transform()
        return {
            "scale": transform.m11(),
            "horizontal_scroll": self.horizontalScrollBar().value(),
            "vertical_scroll": self.verticalScrollBar().value(),
            "translate": (transform.dx(), transform.dy()),
        }

    def deserialize(self, data):
        scale = data.get("scale", 1.0)
        translate = data.get("translate", (0, 0))
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.resetTransform()
        self.translate(translate[0], translate[1])
        self.scale(scale, scale)
        self.horizontalScrollBar().setValue(data.get("horizontal_scroll", 0))
        self.verticalScrollBar().setValue(data.get("vertical_scroll", 0))
        self.setTransformationAnchor(anchor)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)

        # Grid
        left = math.floor(rect.left())
        right = math.ceil(rect.right())
        top = math.floor(rect.top())
        bottom = math.ceil(rect.bottom())

        first_left = left - (left % 64)
        first_top = top - (top % 64)

        lines = []
        for x in range(first_left, right, 64):
            lines.append(QLine(x, top, x, bottom))
        for y in range(first_top, bottom, 64):
            lines.append(QLine(left, y, right, y))

        painter.setPen(QPen(Qt.darkGray, 1, Qt.SolidLine))
        painter.drawLines(lines)

    def drawForeground(self, painter, rect):
        if self._selection_rect is not None:
            painter.setCompositionMode(QPainter.CompositionMode_Difference)
            painter.setPen(QPen(Qt.white, 2, Qt.DashLine))
            painter.drawRect(self._selection_rect)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene().itemAt(scene_pos, QTransform())
            if item is None:
                self._start_pos = scene_pos
                self._selection_rect = QRectF(self._start_pos, self._start_pos)
                self.update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._start_pos is not None:
            self._selection_rect.setBottomRight(self.mapToScene(event.pos()))
            self.update()

        self.target_viewport_pos = event.position().toPoint()
        self.target_scene_pos = self.mapToScene(self.target_viewport_pos)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._start_pos is not None:
            for item in self.scene().items(self._selection_rect):
                item.setSelected(True)
            self._start_pos = None
            self._selection_rect = None
            self.update()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            dy = event.angleDelta().y()
            if dy:
                scale_rate = 1.005
                scale_delta = pow(scale_rate if dy > 0 else 1 / scale_rate, abs(dy))

                if self.target_viewport_pos is None:
                    self.update_target_pos(event)

                self.scale(scale_delta, scale_delta)
                self.centerOn(self.target_scene_pos)
                viewport_center = QPoint(self.viewport().width() / 2, self.viewport().height() / 2)
                delta_viewport_pos = self.target_viewport_pos - viewport_center
                viewport_center = self.mapFromScene(self.target_scene_pos) - delta_viewport_pos
                self.centerOn(self.mapToScene(viewport_center))

        else:
            super().wheelEvent(event)
            self.target_viewport_pos = event.position().toPoint()
            self.target_scene_pos = self.mapToScene(self.target_viewport_pos)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            for item in self.scene().selectedItems():
                self.scene().removeItem(item)
                del item
        else:
            super().keyPressEvent(event)

    def update_target_pos(self, event):
        self.target_viewport_pos = event.position().toPoint()
        self.target_scene_pos = self.mapToScene(self.target_viewport_pos)
