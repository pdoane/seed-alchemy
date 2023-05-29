from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import (
    QGuiApplication,
    QIcon,
    QIconEngine,
    QPainter,
    QPalette,
    QPixmap,
)

from . import configuration
from . import font_awesome as fa


class FontAwesomeIconEngine(QIconEngine):
    def __init__(self, icon_code):
        super().__init__()
        self.icon_code = icon_code

    def paint(self, painter, rect, mode, state):
        palette = QGuiApplication.palette()

        if mode == QIcon.Mode.Disabled:
            color = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        else:
            color = palette.text().color()

        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(fa.pixmap_font)
        painter.setPen(color)

        painter.drawText(rect, Qt.AlignCenter, self.icon_code)

    def pixmap(self, size, mode, state):
        pixmap = QPixmap(size)
        # pixmap.setDevicePixelRatio(QGuiApplication.instance().devicePixelRatio())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        return pixmap


class PixmapIconEngine(QIconEngine):
    def __init__(self, icon_path):
        super().__init__()
        self.icon_path = configuration.get_resource_path(icon_path)

    def paint(self, painter, rect, mode, state):
        palette = QGuiApplication.palette()

        if mode == QIcon.Mode.Disabled:
            color = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        else:
            color = palette.text().color()

        pixmap = QPixmap(self.icon_path)
        colored_pixmap = QPixmap(pixmap.size())
        colored_pixmap.fill(color)

        with QPainter(pixmap) as color_painter:
            color_painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
            color_painter.drawPixmap(0, 0, colored_pixmap)

        painter.drawPixmap(rect, pixmap)

    def pixmap(self, size, mode, state):
        pixmap = QPixmap(size)
        # pixmap.setDevicePixelRatio(QGuiApplication.instance().devicePixelRatio())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        return pixmap
