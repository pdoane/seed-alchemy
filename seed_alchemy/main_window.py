import os

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QMainWindow,
    QMenu,
    QProgressBar,
    QStackedLayout,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from . import actions, configuration
from .about_dialog import AboutDialog
from .backend import Backend
from .canvas_mode import CanvasModeWidget
from .gallery_mode import GalleryModeWidget
from .image_mode import ImageModeWidget
from .interrogate_mode import InterrogateModeWidget
from .preferences_dialog import PreferencesDialog
from .prompt_mode import PromptModeWidget
from .thumbnail_loader import ThumbnailLoader
from .thumbnail_model import ThumbnailModel

IMAGE_MODE = 0
CANVAS_MODE = 1
GALLERY_MODE = 2
PROMPT_MODE = 3
INTERROGATE_MODE = 4


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings, collections: list[str]):
        super().__init__()

        self.mode = "image"
        self.settings = settings
        self.collections = collections
        self.thumbnail_loader = ThumbnailLoader()
        QApplication.instance().aboutToQuit.connect(self.thumbnail_loader.shutdown)
        self.thumbnail_model = ThumbnailModel(self.thumbnail_loader, 100)

        self.setFocusPolicy(Qt.ClickFocus)

        # File Menu
        action_preferences = actions.preferences.create(self)
        action_preferences.triggered.connect(self.show_preferences_dialog)
        action_quit = actions.quit.create(self)
        action_quit.triggered.connect(self.close)

        self.app_menu = QMenu("File", self)
        self.app_menu.addAction(action_preferences)
        self.app_menu.addSeparator()
        self.app_menu.addAction(action_quit)

        # Tools Menu
        convert_model_action = actions.convert_model.create(self)
        convert_model_action.triggered.connect(self.on_convert_model)

        self.tools_menu = QMenu("Tools", self)
        self.tools_menu.addAction(convert_model_action)

        # Help Menu
        action_about = actions.about.create(self)
        action_about.triggered.connect(self.show_about_dialog)

        self.help_menu = QMenu("Help", self)
        self.help_menu.addAction(action_about)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setStyleSheet("QProgressBar { border: none; }")

        # Backend
        self.backend = Backend(self.progress_bar)

        # Modes
        mode_toolbar = QToolBar()
        mode_toolbar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, mode_toolbar)

        image_mode_button = actions.image_mode.mode_button()
        canvas_mode_button = actions.canvas_mode.mode_button()
        gallery_mode_button = actions.gallery_mode.mode_button()
        prompt_mode_button = actions.prompt_mode.mode_button()
        interrogate_mode_button = actions.interrogate_mode.mode_button()

        mode_toolbar.addWidget(image_mode_button)
        mode_toolbar.addWidget(canvas_mode_button)
        mode_toolbar.addWidget(gallery_mode_button)
        mode_toolbar.addWidget(prompt_mode_button)
        mode_toolbar.addWidget(interrogate_mode_button)

        self.button_group = QButtonGroup()
        self.button_group.addButton(image_mode_button, IMAGE_MODE)
        self.button_group.addButton(canvas_mode_button, CANVAS_MODE)
        self.button_group.addButton(gallery_mode_button, GALLERY_MODE)
        self.button_group.addButton(prompt_mode_button, PROMPT_MODE)
        self.button_group.addButton(interrogate_mode_button, INTERROGATE_MODE)
        self.button_group.idToggled.connect(self.on_mode_changed)

        # Stacked Layout
        self.stacked_layout = QStackedLayout()
        self.stacked_layout.addWidget(ImageModeWidget(self))
        self.stacked_layout.addWidget(CanvasModeWidget(self))
        self.stacked_layout.addWidget(GalleryModeWidget(self))
        self.stacked_layout.addWidget(PromptModeWidget(self))
        self.stacked_layout.addWidget(InterrogateModeWidget(self))

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        vlayout = QVBoxLayout(central_widget)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        vlayout.addWidget(self.progress_bar)
        vlayout.addLayout(self.stacked_layout)

        self.setWindowTitle(configuration.APP_NAME)
        self.setGeometry(100, 100, 1200, 600)

        # Update state
        self.set_mode(self.settings.value("mode"))

    def set_mode(self, mode):
        self.mode = mode
        self.settings.setValue("mode", mode)

        if mode == "image":
            index = IMAGE_MODE
        elif mode == "canvas":
            index = CANVAS_MODE
        elif mode == "gallery":
            index = GALLERY_MODE
        elif mode == "prompt":
            index = PROMPT_MODE
        elif mode == "interrogate":
            index = INTERROGATE_MODE

        self.button_group.button(index).setChecked(True)
        self.stacked_layout.setCurrentIndex(index)
        current_widget = self.stacked_layout.currentWidget()

        menu_bar = self.menuBar()
        menu_bar.clear()
        menu_bar.addMenu(self.app_menu)
        for menu in current_widget.get_menus():
            menu_bar.addMenu(menu)
        menu_bar.addMenu(self.tools_menu)
        menu_bar.addMenu(self.help_menu)

        return current_widget

    def on_mode_changed(self, button_id, checked):
        if not checked:
            return
        if button_id == IMAGE_MODE:
            self.set_mode("image")
        elif button_id == CANVAS_MODE:
            self.set_mode("canvas")
        elif button_id == GALLERY_MODE:
            self.set_mode("gallery")
        elif button_id == PROMPT_MODE:
            self.set_mode("prompt")
        elif button_id == INTERROGATE_MODE:
            self.set_mode("interrogate")

    def show_about_dialog(self):
        about_dialog = AboutDialog()
        about_dialog.exec()

    def show_preferences_dialog(self):
        dialog = PreferencesDialog(self)
        dialog.exec()

    def on_convert_model(self):
        dialog = QFileDialog()
        dialog.setNameFilter("Checkpoint files (*.ckpt);;Safetensor files (*.safetensors);;All files (*.*)")
        dialog.selectNameFilter("Checkpoint files (*.ckpt);;Safetensor files (*.safetensors)")

        if dialog.exec() == QFileDialog.Accepted:
            checkpoint_path = dialog.selectedFiles()[0]
            base_name, ext = os.path.splitext(checkpoint_path)
            original_config_file = base_name + ".yaml"
            if not os.path.exists(original_config_file):
                original_config_file = None
            from_safetensors = ext.lower() == ".safetensors"

            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=checkpoint_path,
                original_config_file=original_config_file,
                from_safetensors=from_safetensors,
            )

            os.mkdir(base_name)
            pipe.save_pretrained(base_name, safe_serialization=True)

    def closeEvent(self, event):
        current_widget = self.stacked_layout.currentWidget()
        if current_widget and not current_widget.on_close():
            self.hide()
            event.ignore()
        else:
            event.accept()

    def keyPressEvent(self, event):
        current_widget = self.stacked_layout.currentWidget()
        if not current_widget.on_key_press(event):
            super().keyPressEvent(event)
