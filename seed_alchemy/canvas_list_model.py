from PySide6.QtCore import QAbstractListModel, Qt
from PySide6.QtWidgets import QGraphicsScene


class CanvasListModel(QAbstractListModel):
    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.build_items()
        scene.changed.connect(self.refresh)

    def rowCount(self, parent=None):
        return len(self.items)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            item = self.items[index.row()]
            return item.describe()

    def refresh(self):
        self.beginResetModel()
        self.build_items()
        self.endResetModel()

    def build_items(self):
        self.items = [item for item in self.scene.items() if item.zValue() == 0]
