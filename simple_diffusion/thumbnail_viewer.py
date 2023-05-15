
import os

from PIL import Image
from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QComboBox, QMenu, QVBoxLayout, QWidget

from . import actions, configuration
from .image_metadata import ImageMetadata
from .thumbnail_list_widget import ThumbnailListWidget
from .thumbnail_loader import ThumbnailLoader


class ThumbnailViewer(QWidget):
    file_dropped = Signal(str)

    def __init__(self, loader: ThumbnailLoader, settings: QSettings, collections: list[str], source_image_menu, parent=None):
        super().__init__(parent)

        self.setAcceptDrops(True)

        self.action_use_prompt = actions.use_prompt.create()
        self.action_use_seed = actions.use_seed.create()
        self.action_use_all = actions.use_all.create()
        self.action_use_source_images = actions.use_source_images.create()
        self.action_delete = actions.delete_image.create()
        self.action_reveal_in_finder = actions.reveal_in_finder.create()

        self.collection_combobox = QComboBox()

        self.list_widget = ThumbnailListWidget(loader)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)

        thumbnail_layout = QVBoxLayout(self)
        thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        thumbnail_layout.addWidget(self.collection_combobox)
        thumbnail_layout.addWidget(self.list_widget)

        self.menu = QMenu()
        self.menu.addMenu(source_image_menu)
        self.menu.addSeparator()
        self.menu.addAction(self.action_use_prompt)
        self.menu.addAction(self.action_use_seed)
        self.menu.addAction(self.action_use_source_images)
        self.menu.addAction(self.action_use_all)
        self.menu.addSeparator()
        self.menu.addAction(self.action_delete)
        self.menu.addSeparator()
        self.menu.addAction(self.action_reveal_in_finder)

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

    def get_current_metadata(self):
        rel_path = self.list_widget.selected_image()
        if rel_path:
            full_path = os.path.join(configuration.IMAGES_PATH, rel_path)
            with Image.open(full_path) as image:
                metadata = ImageMetadata()
                metadata.path = rel_path
                metadata.load_from_image(image)
                return metadata
        return None

    def show_context_menu(self, point):
        self.menu.exec(self.mapToGlobal(point))