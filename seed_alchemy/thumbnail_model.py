from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, Qt
from PySide6.QtGui import QIcon

from .thumbnail_loader import ThumbnailLoader


class ThumbnailModel(QAbstractListModel):
    def __init__(self, loader: ThumbnailLoader, icon_size: float):
        super().__init__()
        self.icon_size = icon_size
        self.image_paths = []
        self.referenced_images = {}
        self.loader = loader
        self.cache = {}

    def update_icon_size(self, icon_size):
        self.icon_size = icon_size
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()):
        return len(self.image_paths)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.SizeHintRole:
            return QSize(self.icon_size, self.icon_size)
        if role != Qt.DecorationRole:
            return None

        image_path = self.image_paths[index.row()]
        if image_path not in self.cache:
            self.cache[image_path] = QIcon()
            self.loader.get(image_path, 256, self.set_thumbnail)

        return QIcon(self.cache[image_path])

    def set_thumbnail(self, image_path, pixmap):
        if image_path in self.image_paths:
            row = self.image_paths.index(image_path)
            self.cache[image_path] = pixmap
            self.dataChanged.emit(self.index(row), self.index(row))

    def add_reference(self, image_path):
        if image_path in self.referenced_images:
            self.referenced_images[image_path] += 1
        else:
            self.referenced_images[image_path] = 1

        if image_path in self.image_paths:
            row = self.image_paths.index(image_path)
            self.dataChanged.emit(self.index(row), self.index(row))

    def remove_reference(self, image_path):
        if image_path in self.referenced_images:
            self.referenced_images[image_path] -= 1
            if self.referenced_images[image_path] <= 0:
                del self.referenced_images[image_path]
        elif image_path != "":
            raise ValueError("Image not found in referenced_images")

        if image_path in self.image_paths:
            row = self.image_paths.index(image_path)
            self.dataChanged.emit(self.index(row), self.index(row))
