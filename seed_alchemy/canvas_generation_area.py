from PySide6.QtCore import QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem


class CanvasGenerationArea(QGraphicsRectItem, QObject):
    image_size_changed = Signal(QSize)

    def __init__(self, parent=None):
        QGraphicsRectItem.__init__(self, QRectF(), parent)
        QObject.__init__(self, parent)

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
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
        else:
            self.setRect(QRectF(0, 0, 512, 512))

    def setRect(self, rect):
        old_rect = self.rect()
        super().setRect(rect)
        if old_rect != rect:
            self.image_size_changed.emit(rect.size().toSize())

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

        new_rect = QRectF(self._startRect)

        dx = event.scenePos().x() - self._startPos.x()
        dy = event.scenePos().y() - self._startPos.y()

        dx = round(dx / 8) * 8
        dy = round(dy / 8) * 8

        if self._resize_x == "left":
            new_rect.setLeft(self._startRect.left() + dx)
        elif self._resize_x == "right":
            new_rect.setRight(self._startRect.right() + dx)

        if self._resize_y == "top":
            new_rect.setTop(self._startRect.top() + dy)
        elif self._resize_y == "bottom":
            new_rect.setBottom(self._startRect.bottom() + dy)

        if new_rect.width() < 8 or new_rect.height() < 8:
            return

        self.setRect(new_rect)
        self.image_size_changed.emit(new_rect.size().toSize())

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
