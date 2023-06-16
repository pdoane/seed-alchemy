from PySide6.QtCore import QObject, QRectF, QSize, Qt, Signal, QPointF, QSizeF
from PySide6.QtGui import QColor, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem

HANDLE_SIZE = 5


class CanvasGenerationArea(QGraphicsRectItem, QObject):
    image_size_changed = Signal(QSize)

    def __init__(self, parent=None):
        QGraphicsRectItem.__init__(self, QRectF(), parent)
        QObject.__init__(self, parent)
        self._startPos = None
        self._startRect = None
        self._resize_x = None
        self._resize_y = None
        self._cursor = None

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

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

    def boundingRect(self):
        return self.rect().adjusted(-HANDLE_SIZE, -HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE)

    def shape(self):
        border_path = QPainterPath()
        border_path.addRect(self.rect())

        border_stroker = QPainterPathStroker()
        border_stroker.setWidth(2)
        border_path = border_stroker.createStroke(border_path)

        handles_path = QPainterPath()
        for rect in self._get_handles():
            handles_path.addRect(rect)

        return handles_path.united(border_path)

    def paint(self, painter, option, widget):
        painter.save()
        painter.setPen(QPen(Qt.white, 2))
        painter.drawRect(self.rect())
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        painter.drawRects(self._get_handles())
        painter.restore()

    def mousePressEvent(self, event):
        self._startPos = event.scenePos()
        self._startRect = self.rect()
        cursor, resize_x, resize_y = self._get_resize_info(event.pos())
        self._resize_x = resize_x
        self._resize_y = resize_y
        self._cursor = cursor
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._startPos is None:
            return

        new_rect = QRectF(self._startRect)

        dx = event.scenePos().x() - self._startPos.x()
        dy = event.scenePos().y() - self._startPos.y()

        dx = round(dx / 8) * 8
        dy = round(dy / 8) * 8

        if self._cursor == Qt.OpenHandCursor:
            new_rect.translate(dx, dy)
        else:
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
        self._cursor = None
        self._resize_x = None
        self._resize_y = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event):
        cursor, _, _ = self._get_resize_info(event.pos())
        self.setCursor(cursor)

    def _get_handles(self):
        rect = self.rect()
        offset = QPointF(-HANDLE_SIZE, -HANDLE_SIZE)
        size = QSizeF(HANDLE_SIZE * 2, HANDLE_SIZE * 2)
        return [
            QRectF(rect.topLeft() + offset, size),
            QRectF(rect.topRight() + offset, size),
            QRectF(rect.bottomLeft() + offset, size),
            QRectF(rect.bottomRight() + offset, size),
            QRectF(QPointF(rect.center().x(), rect.top()) + offset, size),
            QRectF(QPointF(rect.center().x(), rect.bottom()) + offset, size),
            QRectF(QPointF(rect.left(), rect.center().y()) + offset, size),
            QRectF(QPointF(rect.right(), rect.center().y()) + offset, size),
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

        outer_rect = self.rect().adjusted(-1, -1, 1, 1)
        inner_rect = self.rect().adjusted(1, 1, -1, -1)
        if outer_rect.contains(pos) and not inner_rect.contains(pos):
            return Qt.OpenHandCursor, None, None

        return Qt.ArrowCursor, None, None
