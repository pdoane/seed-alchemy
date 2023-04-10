from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QListWidget,
                               QStyle, QStyledItemDelegate, QStyleOptionSlider)

class ThumbnailSelectionDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)

        if option.state & QStyle.State_Selected:
            rect = QRect(option.rect)
            checkmark_start = QPoint(rect.left() * 0.70 + rect.right() * 0.30, rect.top() * 0.50 + rect.bottom() * 0.50)
            checkmark_middle = QPoint(rect.left() * 0.55 + rect.right() * 0.45, rect.top() * 0.35 + rect.bottom() * 0.65)
            checkmark_end = QPoint(rect.left() * 0.25 + rect.right() * 0.75, rect.top() * 0.65 + rect.bottom() * 0.35)

            painter.save()
            painter.setPen(QPen(QColor(0, 255, 0), 8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawLine(checkmark_start, checkmark_middle)
            painter.drawLine(checkmark_middle, checkmark_end)
            painter.restore()

class ThumbnailListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_thumbnail_size = 100
        self.max_thumbnail_size = 250
        self.spacing = 8

        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSpacing(self.spacing)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setItemDelegate(ThumbnailSelectionDelegate(self))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_icon_size()

    def visualRect(self, index):
        rect = super().visualRect(index)
        rect.setWidth(self.iconSize().width())
        rect.setHeight(self.iconSize().height())
        return rect

    def update_icon_size(self):
        style = QApplication.instance().style()
        scrollbar_width = style.pixelMetric(QStyle.PM_ScrollBarExtent, QStyleOptionSlider())
        available_width = self.width() - scrollbar_width - 4
        num_columns = int((available_width) / (self.min_thumbnail_size + self.spacing * 2))
        num_columns = max(1, num_columns)
        new_icon_size = int((available_width - num_columns * self.spacing * 2) / num_columns)
        new_icon_size = max(self.min_thumbnail_size, min(new_icon_size, self.max_thumbnail_size))

        self.setIconSize(QSize(new_icon_size, new_icon_size))
