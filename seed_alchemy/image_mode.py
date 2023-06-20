import json
import os
import shutil

from PIL import Image, PngImagePlugin
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QMenu,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import actions, configuration
from . import font_awesome as fa
from . import utils
from .control_net_widget import ControlNetWidget
from .delete_image_dialog import DeleteImageDialog
from .generate_thread import GenerateImageTask
from .icon_engine import FontAwesomeIconEngine
from .image_generation_panel import ImageGenerationPanel
from .image_history import ImageHistory
from .image_metadata import ImageMetadata
from .image_viewer import ImageViewer
from .pipelines import GenerateRequest
from .preprocess_task import PreprocessTask
from .processors import ProcessorBase
from .thumbnail_model import ThumbnailModel
from .thumbnail_viewer import ThumbnailViewer


class ImageModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.backend = main_window.backend
        self.settings: QSettings = main_window.settings
        self.collections = main_window.collections
        self.thumbnail_model: ThumbnailModel = main_window.thumbnail_model

        self.image_history = ImageHistory()
        self.image_history.current_image_changed.connect(self.on_current_image_changed)

        self.preview_preprocessor: ProcessorBase = None
        self.preprocess_task = None
        self.generate_task = None

        # Set as Source Menu
        self.current_image_set_as_source_menu = QMenu("Set as Source", self)
        self.current_image_set_as_source_menu.setIcon(QIcon(FontAwesomeIconEngine(fa.icon_share)))
        self.current_image_set_as_source_menu.addAction(QAction("Dummy...", self))
        self.current_image_set_as_source_menu.aboutToShow.connect(self.populate_current_image_set_as_source_menu)

        # Move To Menu
        self.move_image_menu = QMenu("Move To", self)
        self.move_image_menu.setIcon(utils.empty_qicon())
        for collection in self.collections:
            item_action = QAction(collection, self)
            item_action.triggered.connect(lambda checked=False, collection=collection: self.on_move_image(collection))
            self.move_image_menu.addAction(item_action)

        # Generation Panel
        self.generation_panel = ImageGenerationPanel(main_window, "image")
        self.generation_panel.generate_requested.connect(self.generate_requested)
        self.generation_panel.cancel_requested.connect(self.cancel_requested)
        self.generation_panel.preprocess_requested.connect(self.preprocess_requested)
        self.generation_panel.locate_source_requested.connect(self.locate_source_requested)

        # Image viewer
        self.image_viewer = ImageViewer(self.image_history)
        self.image_viewer.set_as_source_image_button.setMenu(self.current_image_set_as_source_menu)
        self.image_viewer.use_prompt_button.clicked.connect(self.on_use_prompt)
        self.image_viewer.use_seed_button.clicked.connect(self.on_use_seed)
        self.image_viewer.use_source_images_button.clicked.connect(self.on_use_source_images)
        self.image_viewer.use_all_button.clicked.connect(self.on_use_all)
        self.image_viewer.delete_button.clicked.connect(self.on_delete)

        image_viewer_frame = QFrame()
        image_viewer_frame.setFrameShape(QFrame.Panel)

        image_viewer_layout = QVBoxLayout(image_viewer_frame)
        image_viewer_layout.setContentsMargins(0, 0, 0, 0)
        image_viewer_layout.addWidget(self.image_viewer)

        # Thumbnail viewer
        thumbnail_use_prompt_action = actions.use_prompt.create(self)
        thumbnail_use_prompt_action.triggered.connect(self.on_use_prompt)
        thumbnail_use_seed_action = actions.use_seed.create(self)
        thumbnail_use_seed_action.triggered.connect(self.on_use_seed)
        thumbnail_use_source_images_action = actions.use_source_images.create(self)
        thumbnail_use_source_images_action.triggered.connect(self.on_use_source_images)
        thumbnail_use_all_action = actions.use_all.create(self)
        thumbnail_use_all_action.triggered.connect(self.on_use_all)
        thumbnail_delete_action = actions.delete_image.create(self)
        thumbnail_delete_action.triggered.connect(self.on_delete)
        thumbnail_reveal_in_finder_action = actions.reveal_in_finder.create(self)
        thumbnail_reveal_in_finder_action.triggered.connect(self.on_reveal_in_finder)

        thumbnail_menu = QMenu(self)
        thumbnail_menu.addMenu(self.current_image_set_as_source_menu)
        thumbnail_menu.addSeparator()
        thumbnail_menu.addAction(thumbnail_use_prompt_action)
        thumbnail_menu.addAction(thumbnail_use_seed_action)
        thumbnail_menu.addAction(thumbnail_use_source_images_action)
        thumbnail_menu.addAction(thumbnail_use_all_action)
        thumbnail_menu.addSeparator()
        thumbnail_menu.addMenu(self.move_image_menu)
        thumbnail_menu.addAction(thumbnail_delete_action)
        thumbnail_menu.addSeparator()
        thumbnail_menu.addAction(thumbnail_reveal_in_finder_action)

        self.thumbnail_viewer = ThumbnailViewer(self.thumbnail_model, self.settings, self.collections, thumbnail_menu)
        self.thumbnail_viewer.file_dropped.connect(self.on_thumbnail_file_dropped)
        self.thumbnail_viewer.list_widget.image_selection_changed.connect(self.on_thumbnail_selection_change)

        # Image Mode
        image_mode_splitter = QSplitter(Qt.Orientation.Horizontal)
        image_mode_splitter.addWidget(image_viewer_frame)
        image_mode_splitter.addWidget(self.thumbnail_viewer)
        image_mode_splitter.setStretchFactor(0, 1)  # left widget
        image_mode_splitter.setStretchFactor(1, 0)  # right widget

        image_mode_layout = QHBoxLayout(self)
        image_mode_layout.setContentsMargins(8, 2, 8, 8)
        image_mode_layout.setSpacing(8)
        image_mode_layout.addWidget(self.generation_panel)
        image_mode_layout.addWidget(image_mode_splitter)

        # Deserialize
        self.generation_panel.deserialize(json.loads(self.settings.value("generation", "{}")))

        # Sync panels
        selected_image = self.thumbnail_viewer.list_widget.selected_image()
        if selected_image is not None:
            self.image_history.visit(selected_image)

    def get_menus(self):
        # History Menu
        history_menu = QMenu("History", self)
        history_menu.addAction(QAction("Dummy...", self))
        history_menu.aboutToShow.connect(lambda: self.image_history.populate_history_menu(history_menu))

        # Prompt Menu
        insert_textual_inversion_action = actions.insert_textual_inversion.create(self)
        insert_textual_inversion_action.triggered.connect(self.on_insert_textual_inversion)
        insert_lora_action = actions.insert_lora.create(self)
        insert_lora_action.triggered.connect(self.on_insert_lora)

        prompt_menu = QMenu("Prompt", self)
        prompt_menu.addAction(insert_textual_inversion_action)
        prompt_menu.addAction(insert_lora_action)

        # Image Menu
        action_generate_image = actions.generate_image.create(self)
        action_generate_image.triggered.connect(self.on_generate_image)
        action_cancel_generation = actions.cancel_generation.create(self)
        action_cancel_generation.triggered.connect(self.on_cancel_generation)
        action_use_prompt = actions.use_prompt.create(self)
        action_use_prompt.triggered.connect(self.on_use_prompt)
        action_use_seed = actions.use_seed.create(self)
        action_use_seed.triggered.connect(self.on_use_seed)
        action_use_source_images = actions.use_source_images.create(self)
        action_use_source_images.triggered.connect(self.on_use_source_images)
        action_use_all = actions.use_all.create(self)
        action_use_all.triggered.connect(self.on_use_all)
        action_toggle_metadata = actions.toggle_metadata.create(self)
        action_toggle_metadata.triggered.connect(lambda: self.image_viewer.toggle_metadata_button.toggle())
        action_toggle_preview = actions.toggle_preview.create(self)
        action_toggle_preview.triggered.connect(lambda: self.image_viewer.toggle_preview_button.toggle())
        action_delete_image = actions.delete_image.create(self)
        action_delete_image.triggered.connect(self.on_delete)
        action_reveal_in_finder = actions.reveal_in_finder.create(self)
        action_reveal_in_finder.triggered.connect(self.on_reveal_in_finder)

        image_menu = QMenu("Image", self)
        image_menu.addAction(action_generate_image)
        image_menu.addAction(action_cancel_generation)
        image_menu.addSeparator()
        image_menu.addMenu(self.current_image_set_as_source_menu)
        image_menu.addSeparator()
        image_menu.addAction(action_use_prompt)
        image_menu.addAction(action_use_seed)
        image_menu.addAction(action_use_source_images)
        image_menu.addAction(action_use_all)
        image_menu.addSeparator()
        image_menu.addAction(action_toggle_metadata)
        image_menu.addAction(action_toggle_preview)
        image_menu.addSeparator()
        image_menu.addMenu(self.move_image_menu)
        image_menu.addAction(action_delete_image)
        image_menu.addSeparator()
        image_menu.addAction(action_reveal_in_finder)
        image_menu.addSeparator()

        return [history_menu, prompt_menu, image_menu]

    def populate_current_image_set_as_source_menu(self):
        self.current_image_set_as_source_menu.clear()
        self.generation_panel.build_set_as_source_menu(
            self.current_image_set_as_source_menu, self.image_viewer.metadata
        )

    def generate_requested(self):
        if self.generate_task:
            return

        params = self.generation_panel.serialize()

        self.settings.setValue("generation", json.dumps(params))
        self.settings.setValue("collection", self.thumbnail_viewer.collection())

        req = GenerateRequest()
        req.collection = self.settings.value("collection")
        req.reduce_memory = self.settings.value("reduce_memory", type=bool)
        req.image_metadata = ImageMetadata()
        req.image_metadata.load_from_params(params)
        req.num_images_per_prompt = params["num_images_per_prompt"]

        self.generation_panel.begin_generate()
        self.generate_task = GenerateImageTask(req)
        self.generate_task.task_progress.connect(self.update_progress)
        self.generate_task.image_preview.connect(self.image_preview)
        self.generate_task.image_complete.connect(self.image_complete)
        self.generate_task.completed.connect(self.generate_complete)
        self.backend.start(self.generate_task)

    def cancel_requested(self):
        if self.generate_task:
            self.generate_task.cancel = True

    def update_progress(self, progress_amount):
        self.backend.update_progress(progress_amount)

    def image_preview(self, preview_image):
        self.image_viewer.set_preview_image(preview_image)

    def image_complete(self, output_path):
        self.on_add_file(output_path)

    def generate_complete(self):
        self.generation_panel.end_generate()
        self.image_viewer.set_preview_image(None)
        self.generate_task = None

    def preprocess_requested(self, control_net_widget: ControlNetWidget):
        source_path = control_net_widget.get_source_path()
        image_size = self.generation_panel.get_image_size().toTuple()

        full_path = os.path.join(configuration.IMAGES_PATH, source_path)
        with Image.open(full_path) as image:
            image = image.convert("RGB")
            image = image.resize(image_size, Image.Resampling.LANCZOS)
            source_image = image.copy()

        preprocessor_name = control_net_widget.get_preprocessor_name()
        params = control_net_widget.get_param_values()

        self.preprocess_task = PreprocessTask(
            source_image=source_image,
            preprocessor_name=preprocessor_name,
            params=params,
        )
        self.preprocess_task.image_completed.connect(self.preprocess_complete)
        self.backend.start(self.preprocess_task)

    def preprocess_complete(self, image):
        output_path = os.path.join(configuration.TMP_DIR, configuration.PREPROCESSED_IMAGE_NAME)
        full_path = os.path.join(configuration.IMAGES_PATH, output_path)
        image.save(full_path)
        self.image_viewer.set_current_image(output_path)

    def locate_source_requested(self, image_path):
        self.thumbnail_viewer.select_image(image_path)

    def on_current_image_changed(self, path):
        self.image_viewer.set_current_image(path)
        self.thumbnail_viewer.list_widget.image_selection_changed.disconnect()
        self.thumbnail_viewer.select_image(path)
        self.thumbnail_viewer.list_widget.image_selection_changed.connect(self.on_thumbnail_selection_change)

    def on_thumbnail_file_dropped(self, source_path: str):
        collection = self.thumbnail_viewer.collection()

        def io_operation():
            next_image_id = utils.next_image_id(os.path.join(configuration.IMAGES_PATH, collection))
            output_path = os.path.join(collection, "{:05d}.png".format(next_image_id))
            full_path = os.path.join(configuration.IMAGES_PATH, output_path)

            with Image.open(source_path) as image:
                metadata = ImageMetadata()
                metadata.path = output_path
                metadata.load_from_image(image)
                png_info = PngImagePlugin.PngInfo()
                metadata.save_to_png_info(png_info)

                image.save(full_path, pnginfo=png_info)
            return output_path

        output_path = utils.retry_on_failure(io_operation)

        self.on_add_file(output_path)

    def on_thumbnail_selection_change(self, image_path):
        self.image_history.visit(image_path)

    def on_insert_textual_inversion(self):
        self.generation_panel.prompt_edit.on_insert_textual_inversion()

    def on_insert_lora(self):
        self.generation_panel.prompt_edit.on_insert_lora()

    def on_generate_image(self):
        self.generation_panel.on_generate_clicked()

    def on_cancel_generation(self):
        self.generation_panel.on_cancel_clicked()

    def on_use_prompt(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            self.generation_panel.begin_update()
            self.generation_panel.use_prompt(image_metadata)
            self.generation_panel.end_update()

    def on_use_seed(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            self.generation_panel.begin_update()
            self.generation_panel.use_seed(image_metadata)
            self.generation_panel.end_update()

    def on_use_general(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            self.generation_panel.begin_update()
            self.generation_panel.use_general(image_metadata)
            self.generation_panel.end_update()

    def on_use_source_images(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            self.generation_panel.begin_update()
            self.generation_panel.use_source_images(image_metadata)
            self.generation_panel.end_update()

    def on_use_img2img(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            self.generation_panel.begin_update()
            self.generation_panel.use_img2img(image_metadata)
            self.generation_panel.end_update()

    def on_use_control_net(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            self.generation_panel.begin_update()
            self.generation_panel.use_control_net(image_metadata)
            self.generation_panel.end_update()

    def on_use_post_processing(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            self.generation_panel.begin_update()
            self.generation_panel.use_post_processing(image_metadata)
            self.generation_panel.end_update()

    def on_use_all(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            self.generation_panel.begin_update()
            self.generation_panel.use_all(image_metadata)
            self.generation_panel.end_update()

    def on_move_image(self, collection: str):
        current_collection = self.thumbnail_viewer.collection()
        if collection == current_collection:
            return

        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            source_path = image_metadata.path

            def io_operation():
                next_image_id = utils.next_image_id(os.path.join(configuration.IMAGES_PATH, collection))
                output_path = os.path.join(collection, "{:05d}.png".format(next_image_id))

                full_source_path = os.path.join(configuration.IMAGES_PATH, source_path)
                full_output_path = os.path.join(configuration.IMAGES_PATH, output_path)

                shutil.move(full_source_path, full_output_path)
                return output_path

            output_path = utils.retry_on_failure(io_operation)

            self.on_remove_file(image_metadata.path)
            self.on_add_file(output_path)

    def on_delete(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            full_path = os.path.join(configuration.IMAGES_PATH, image_metadata.path)
            dialog = DeleteImageDialog(full_path)
            focused_widget = QApplication.focusWidget()
            result = dialog.exec()
            if result == QDialog.Accepted:
                utils.recycle_file(full_path)
                self.on_remove_file(image_metadata.path)
            if focused_widget:
                focused_widget.setFocus()

    def on_reveal_in_finder(self):
        image_metadata = self.image_viewer.metadata
        if image_metadata is not None:
            full_path = os.path.abspath(os.path.join(configuration.IMAGES_PATH, image_metadata.path))
            utils.reveal_in_finder(full_path)

    def on_add_file(self, path):
        collection = os.path.dirname(path)
        current_collection = self.thumbnail_viewer.collection()
        if collection != current_collection:
            return

        self.thumbnail_viewer.add_image(path)
        self.image_history.visit(path)

    def on_remove_file(self, path):
        self.thumbnail_viewer.remove_image(path)

    def on_close(self):
        if self.generate_task:
            self.generate_task.cancel = True
        return True

    def on_key_press(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self.thumbnail_viewer.previous_image()
        elif key == Qt.Key_Right:
            self.thumbnail_viewer.next_image()
        else:
            return False
        return True
