
import os

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QComboBox, QMenu, QVBoxLayout, QWidget

from . import configuration
from .thumbnail_list_widget import ThumbnailListWidget
from .thumbnail_loader import ThumbnailLoader


class ThumbnailViewer(QWidget):
    file_dropped = Signal(str)

    def __init__(self, loader: ThumbnailLoader, settings: QSettings, collections: list[str], context_menu: QMenu, parent=None):
        super().__init__(parent)

        self.setAcceptDrops(True)

        self.collection_combobox = QComboBox()

        self.list_widget = ThumbnailListWidget(loader)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)

        thumbnail_layout = QVBoxLayout(self)
        thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        thumbnail_layout.addWidget(self.collection_combobox)
        thumbnail_layout.addWidget(self.list_widget)

        self.menu = context_menu

        # Gather collections
        self.collection_combobox.addItems(collections)
        self.collection_combobox.setCurrentText(settings.value('collection'))
        self.collection_combobox.currentIndexChanged.connect(self.update_collection)
        self.pending_selection = None
        self.update_collection()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet('QListWidget {background-color: #222233;}')

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.setStyleSheet('QListWidget {background-color: black;}')

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            self.file_dropped.emit(url.toLocalFile())
        self.setStyleSheet('QListWidget {background-color: black;}')

        event.acceptProposedAction()

    def collection(self):
        return self.collection_combobox.currentText()

    def update_collection(self):
        collection = self.collection()

        image_files = sorted([
            file for file in os.listdir(os.path.join(configuration.IMAGES_PATH, collection))
            if file.lower().endswith(('.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
        image_files.reverse()

        self.list_widget.clear()

        for image_file in image_files:
            image_path = os.path.join(collection, image_file)
            self.list_widget.add_image(image_path)

        if self.pending_selection:
            self.list_widget.select_image(self.pending_selection)
            self.pending_selection = None
        else:
            self.list_widget.select_index(0)

    def select_image(self, rel_path):
        collection = os.path.dirname(rel_path)
        if self.collection() != collection:
            self.pending_selection = rel_path
            self.collection_combobox.setCurrentText(collection)
        else:
            self.list_widget.select_image(rel_path)
    
    def add_image(self, rel_path):
        self.list_widget.insert_image(0, rel_path)
        self.list_widget.select_index(0, scroll_to=False)

    def remove_image(self, rel_path):
        self.list_widget.remove_image(rel_path)

    def previous_image(self):
        next_row = self.list_widget.selected_index() - 1
        if next_row >= 0:
            self.list_widget.select_index(next_row)

    def next_image(self):
        next_row = self.list_widget.selected_index() + 1
        if next_row < self.list_widget.count():
            self.list_widget.select_index(next_row)

    def show_context_menu(self, point):
        self.menu.exec(self.list_widget.mapToGlobal(point))
