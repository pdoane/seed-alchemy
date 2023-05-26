from PySide6.QtCore import (
    QItemSelectionModel,
    QModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QListWidget,
    QStyle,
    QStyledItemDelegate,
)

from .thumbnail_model import ThumbnailModel


class ThumbnailSelectionDelegate(QStyledItemDelegate):
    def __init__(self, model: ThumbnailModel, parent=None):
        super().__init__(parent)
        self.model = model

    def paint(self, painter, option, index):
        super().paint(painter, option, index)

        if option.state & QStyle.State_Selected:
            rect = QRect(option.rect)
            checkmark_start = QPoint(
                rect.left() * 0.70 + rect.right() * 0.30, rect.top() * 0.50 + rect.bottom() * 0.50
            )
            checkmark_middle = QPoint(
                rect.left() * 0.55 + rect.right() * 0.45, rect.top() * 0.35 + rect.bottom() * 0.65
            )
            checkmark_end = QPoint(rect.left() * 0.25 + rect.right() * 0.75, rect.top() * 0.65 + rect.bottom() * 0.35)

            painter.save()
            painter.setPen(QPen(QColor(0, 255, 0), 8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawLine(checkmark_start, checkmark_middle)
            painter.drawLine(checkmark_middle, checkmark_end)
            painter.restore()

        image_path = self.model.image_paths[index.row()]
        if image_path in self.model.referenced_images:
            rect = QRect(option.rect)

            painter.save()
            painter.setPen(QPen(QColor(127, 0, 0), 8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawRect(QRect(rect.right() - 12, rect.top() + 4, 8, 8))
            painter.restore()


class ThumbnailListWidget(QListView):
    image_selection_changed = Signal(str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.base_icon_size = model.icon_size
        self.spacing = 8

        self.setUniformItemSizes(True)
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSpacing(self.spacing)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setItemDelegate(ThumbnailSelectionDelegate(self.model, self))
        self.setModel(self.model)
        self.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.viewport().width()
        if not self.verticalScrollBar().isVisible():
            width -= self.verticalScrollBar().width()
        num_columns = max(1, width // (self.base_icon_size + self.spacing * 2))
        icon_size = max(self.base_icon_size, (width - num_columns * self.spacing * 2) // num_columns)
        self.setIconSize(QSize(icon_size, icon_size))
        self.model.update_icon_size(icon_size)

    def clear(self):
        self.model.beginRemoveRows(QModelIndex(), 0, len(self.model.image_paths) - 1)
        self.model.image_paths.clear()
        self.model.endRemoveRows()
        self.model.cache.clear()

    def count(self):
        return len(self.model.image_paths)

    def add_image(self, path):
        row = len(self.model.image_paths)
        self.model.beginInsertRows(QModelIndex(), row, row)
        self.model.image_paths.append(path)
        self.model.endInsertRows()

    def insert_image(self, row, path):
        self.model.beginInsertRows(QModelIndex(), row, row)
        self.model.image_paths.insert(row, path)
        self.model.endInsertRows()

    def remove_image(self, path):
        if path in self.model.image_paths:
            row = self.model.image_paths.index(path)
            self.model.beginRemoveRows(QModelIndex(), row, row)
            self.model.image_paths.pop(row)
            self.model.endRemoveRows()
            self.model.cache.pop(path, None)

    def select_image(self, path, scroll_to=True):
        if path in self.model.image_paths:
            row = self.model.image_paths.index(path)
            index = self.model.index(row, 0)
            scrollbar_position = self.verticalScrollBar().value()
            self.selectionModel().setCurrentIndex(index, QItemSelectionModel.ClearAndSelect)
            if not scroll_to:
                self.verticalScrollBar().setValue(scrollbar_position)

    def select_index(self, row, scroll_to=True):
        index = self.model.index(row, 0)
        if index.isValid():
            scrollbar_position = self.verticalScrollBar().value()
            self.selectionModel().setCurrentIndex(index, QItemSelectionModel.ClearAndSelect)
            if not scroll_to:
                self.verticalScrollBar().setValue(scrollbar_position)

    def selected_image(self):
        selected_indexes = self.selectedIndexes()
        if selected_indexes:
            selected_index = selected_indexes[0]
            image_path = self.model.image_paths[selected_index.row()]
            return image_path
        return None

    def selected_index(self):
        selected_indexes = self.selectedIndexes()
        if selected_indexes:
            return selected_indexes[0].row()
        return -1

    def on_selection_changed(self):
        image_path = self.selected_image()
        if image_path:
            self.image_selection_changed.emit(image_path)
