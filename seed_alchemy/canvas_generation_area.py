from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem


class CanvasGenerationArea(QGraphicsRectItem):
    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setPen(QPen(QColor(Qt.white), 2))

        self._startPos = None
        self._startRect = None
        self._resize_x = None
        self._resize_y = None

    def serialize(self):
        return {
            "class": self.__class__.__name__,
            "rect": (self.rect().left(), self.rect().top(), self.rect().width(), self.rect().height()),
        }

    def deserialize(self, data):
        if "rect" in data:
            self.setRect(QRectF(*data["rect"]))

    def shape(self):
        rect = self.boundingRect().adjusted(4, 4, -4, -4)
        path = QPainterPath()
        path.addRect(rect)

        stroker = QPainterPathStroker()
        stroker.setWidth(8)
        return stroker.createStroke(path)

    def mousePressEvent(self, event):
        self._startPos = event.scenePos()
        self._startRect = self.rect()
        _, resize_x, resize_y = self._get_resize_info(event.pos())
        self._resize_x = resize_x
        self._resize_y = resize_y
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._startPos is None:
            return

        newRect = QRectF(self._startRect)

        dx = event.scenePos().x() - self._startPos.x()
        dy = event.scenePos().y() - self._startPos.y()

        dx = round(dx / 8) * 8
        dy = round(dy / 8) * 8

        if self._resize_x == "left":
            newRect.setLeft(self._startRect.left() + dx)
        elif self._resize_x == "right":
            newRect.setRight(self._startRect.right() + dx)

        if self._resize_y == "top":
            newRect.setTop(self._startRect.top() + dy)
        elif self._resize_y == "bottom":
            newRect.setBottom(self._startRect.bottom() + dy)

        if newRect.width() < 8 or newRect.height() < 8:
            return

        self.setRect(newRect)

    def mouseReleaseEvent(self, event):
        self._startPos = None
        self._resizeDirection = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event):
        cursor, _, _ = self._get_resize_info(event.pos())
        self.setCursor(cursor)

    def _get_resize_info(self, pos):
        rect = self.rect()
        inset = 8
        if pos.x() < rect.left() + inset:
            if pos.y() < rect.top() + inset:
                return Qt.SizeFDiagCursor, "left", "top"
            elif pos.y() > rect.bottom() - inset:
                return Qt.SizeBDiagCursor, "left", "bottom"
            else:
                return Qt.SizeHorCursor, "left", None
        elif pos.x() > rect.right() - inset:
            if pos.y() < rect.top() + inset:
                return Qt.SizeBDiagCursor, "right", "top"
            elif pos.y() > rect.bottom() - inset:
                return Qt.SizeFDiagCursor, "right", "bottom"
            else:
                return Qt.SizeHorCursor, "right", None
        elif pos.y() < rect.top() + inset:
            return Qt.SizeVerCursor, None, "top"
        elif pos.y() > rect.bottom() - inset:
            return Qt.SizeVerCursor, None, "bottom"
        else:
            return Qt.ArrowCursor, None, None
