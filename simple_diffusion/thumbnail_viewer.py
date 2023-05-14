
import os

from PIL import Image
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QComboBox, QListWidgetItem, QMenu, QVBoxLayout,
                               QWidget)

from . import actions, configuration, utils
from .image_metadata import ImageMetadata
from .thumbnail_list_widget import ThumbnailListWidget


class ThumbnailViewer(QWidget):
    file_dropped = Signal(str)

    def __init__(self, settings: QSettings, collections: list[str], source_image_menu, parent=None):
        super().__init__(parent)

        self.setAcceptDrops(True)

        self.action_use_prompt = actions.use_prompt.create()
        self.action_use_seed = actions.use_seed.create()
        self.action_use_all = actions.use_all.create()
        self.action_use_source_images = actions.use_source_images.create()
        self.action_delete = actions.delete_image.create()
        self.action_reveal_in_finder = actions.reveal_in_finder.create()

        self.collection_combobox = QComboBox()

        self.list_widget = ThumbnailListWidget()
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

        os.makedirs(os.path.join(configuration.THUMBNAILS_PATH, collection), exist_ok=True)
        image_files = sorted([file for file in os.listdir(os.path.join(configuration.IMAGES_PATH, collection)) if file.lower().endswith(('.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self.list_widget.blockSignals(False)

        for image_file in image_files:
            image_path = os.path.join(collection, image_file)
            self.add_image(image_path)

        self.select_index_no_scroll(0)

    def select_image(self, rel_path):
        collection = os.path.dirname(rel_path)
        collection_path = os.path.join(configuration.IMAGES_PATH, rel_path)
        if os.path.exists(collection_path):
            self.collection_combobox.setCurrentText(collection)

            for index in range(self.list_widget.count()):
                item = self.list_widget.item(index)
                if item.data(Qt.UserRole) == rel_path:
                    self.list_widget.setCurrentItem(item)
                    break
    
    def select_index_no_scroll(self, index):
        scrollbar_position = self.list_widget.verticalScrollBar().value()
        self.list_widget.setCurrentRow(index)
        self.list_widget.verticalScrollBar().setValue(scrollbar_position)

    def add_image(self, rel_path):
        thumbnail_path = os.path.join(configuration.THUMBNAILS_PATH, rel_path)
        if not os.path.exists(thumbnail_path):
            image_path = os.path.join(configuration.IMAGES_PATH, rel_path)
            with Image.open(image_path) as image:
                thumbnail_image = utils.create_thumbnail(image)
                thumbnail_image.save(thumbnail_path, 'WEBP')

        with Image.open(thumbnail_path) as image:
            pixmap = QPixmap.fromImage(utils.pil_to_qimage(image))
            icon = QIcon(pixmap)
            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(Qt.UserRole, rel_path)
            item.setSizeHint(self.list_widget.iconSize())
            self.list_widget.insertItem(0, item)

    def remove_image(self, rel_path):
        scrollbar_position = self.list_widget.verticalScrollBar().value()
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.data(Qt.UserRole) == rel_path:
                self.list_widget.takeItem(index)
                break
        self.list_widget.verticalScrollBar().setValue(scrollbar_position)

    def previous_image(self):
        next_row = self.list_widget.currentRow() - 1
        if next_row >= 0:
            self.list_widget.setCurrentRow(next_row)

    def next_image(self):
        next_row = self.list_widget.currentRow() + 1
        if next_row < self.list_widget.count():
            self.list_widget.setCurrentRow(next_row)

    def get_current_metadata(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            item = selected_items[0]
            rel_path = item.data(Qt.UserRole)
            full_path = os.path.join(configuration.IMAGES_PATH, rel_path)
            with Image.open(full_path) as image:
                metadata = ImageMetadata()
                metadata.path = rel_path
                metadata.load_from_image(image)
                return metadata
        return None

    def show_context_menu(self, point):
        self.menu.exec(self.mapToGlobal(point))