import json
import os
import random
import shutil
import sys

import actions
import configuration
import font_awesome as fa
import utils
from about_dialog import AboutDialog
from delete_image_dialog import DeleteImageDialog
from generate_thread import GenerateThread
from image_metadata import ControlNetMetadata, ImageMetadata
from image_viewer import ImageViewer
from PIL import Image, PngImagePlugin
from preferences_dialog import PreferencesDialog
from processors import ProcessorBase
from prompt_text_edit import PromptTextEdit
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDialog,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMenu, QMenuBar,
                               QProgressBar, QPushButton, QSizePolicy,
                               QSplitter, QToolBar, QVBoxLayout, QWidget)
from thumbnail_viewer import ThumbnailViewer
from widgets import (ComboBox, DoubleSpinBox, FloatSliderSpinBox, ScrollArea,
                     SpinBox)

if sys.platform == 'darwin':
    from AppKit import NSApplication

class SourceImageUI:
    frame: QFrame = None
    label: QLabel = None
    line_edit: QLineEdit = None

class ImageRefUI:
    frame: QFrame = None
    combo_box: ComboBox = None

class ControlNetFrame(QFrame):
    model_combo_box: ComboBox = None
    image_ref_ui: ImageRefUI = None
    preprocess_check_box: QCheckBox = None
    scale: FloatSliderSpinBox = None

class MainWindow(QMainWindow):
    preview_preprocessor: ProcessorBase = None
    source_image_uis: list[SourceImageUI] = []
    current_source_image_ui: SourceImageUI = None
    img2img_ref_ui: ImageRefUI = None
    control_net_frames: list[ControlNetFrame] = []

    def __init__(self, settings: QSettings, collections: list[str]):
        super().__init__()

        self.settings = settings
        self.collections = collections

        self.generate_thread = None
        self.active_thread_count = 0

        self.setFocusPolicy(Qt.ClickFocus)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Menubar
        menu_bar = QMenuBar(self)

        action_about = actions.about.create(self)
        action_preferences = actions.preferences.create(self)

        action_generate_image = actions.generate_image.create(self)
        action_cancel_generation = actions.cancel_generation.create(self)
        action_send_to_img2img = actions.send_to_img2img.create(self)
        action_use_prompt = actions.use_prompt.create(self)
        action_use_seed = actions.use_seed.create(self)
        action_use_source_images = actions.use_source_images.create(self)
        action_use_all = actions.use_all.create(self)
        action_toggle_metadata = actions.toggle_metadata.create(self)
        action_toggle_preview = actions.toggle_preview.create(self)
        action_delete_image = actions.delete_image.create(self)
        action_reveal_in_finder = actions.reveal_in_finder.create(self)

        move_image_menu = QMenu('Move To', self)
        move_image_menu.setIcon(utils.empty_qicon())
        for collection in self.collections:
            item_action = QAction(collection, self)
            item_action.triggered.connect(lambda checked=False, x=collection: self.on_move_image(self.image_viewer.metadata, x))
            move_image_menu.addAction(item_action)

        app_menu = QMenu("Application", self)
        menu_bar.addMenu(app_menu)
        app_menu.addAction(action_about)
        app_menu.addSeparator()
        app_menu.addAction(action_preferences)

        image_menu = QMenu("Image", menu_bar)
        image_menu.addAction(action_generate_image)
        image_menu.addAction(action_cancel_generation)
        image_menu.addSeparator()
        image_menu.addAction(action_send_to_img2img)
        image_menu.addSeparator()
        image_menu.addAction(action_use_prompt)
        image_menu.addAction(action_use_seed)
        image_menu.addAction(action_use_source_images)
        image_menu.addAction(action_use_all)
        image_menu.addSeparator()
        image_menu.addAction(action_toggle_metadata)
        image_menu.addAction(action_toggle_preview)
        image_menu.addSeparator()
        image_menu.addMenu(move_image_menu)
        image_menu.addAction(action_delete_image)
        image_menu.addSeparator()
        image_menu.addAction(action_reveal_in_finder)
        image_menu.addSeparator()

        action_about.triggered.connect(self.show_about_dialog)
        action_preferences.triggered.connect(self.show_preferences_dialog)

        action_generate_image.triggered.connect(self.on_generate_image)
        action_cancel_generation.triggered.connect(self.on_cancel_generation)
        action_send_to_img2img.triggered.connect(lambda: self.on_send_to_img2img(self.image_viewer.metadata))
        action_use_prompt.triggered.connect(lambda: self.on_use_prompt(self.image_viewer.metadata))
        action_use_seed.triggered.connect(lambda: self.on_use_seed(self.image_viewer.metadata))
        action_use_source_images.triggered.connect(lambda: self.on_use_source_images(self.image_viewer.metadata))
        action_use_all.triggered.connect(lambda: self.on_use_all(self.image_viewer.metadata))
        action_toggle_metadata.triggered.connect(lambda: self.image_viewer.toggle_metadata_button.toggle())
        action_toggle_preview.triggered.connect(lambda: self.image_viewer.toggle_preview_button.toggle())
        action_delete_image.triggered.connect(lambda: self.on_delete(self.image_viewer.metadata))
        action_reveal_in_finder.triggered.connect(lambda: self.on_reveal_in_finder(self.image_viewer.metadata))

        # Add the menu to the menu bar
        menu_bar.addMenu(image_menu)

        # Set the menu bar to the main window
        self.setMenuBar(menu_bar)

        # Modes
        mode_toolbar = QToolBar()
        mode_toolbar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, mode_toolbar)

        image_mode_button = actions.image_mode.tool_button()

        mode_toolbar.addWidget(image_mode_button)

        self.button_group = QButtonGroup()
        self.button_group.addButton(image_mode_button, 0)
        self.button_group.idToggled.connect(self.on_mode_changed)

        # Generate
        self.generate_button = QPushButton('Generate')
        self.generate_button.clicked.connect(self.on_generate_image)

        cancel_button = actions.cancel_generation.push_button()
        cancel_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cancel_button.clicked.connect(self.on_cancel_generation)

        generate_hlayout = QHBoxLayout()
        generate_hlayout.setContentsMargins(0, 0, 0, 0)
        generate_hlayout.setSpacing(2)
        generate_hlayout.addWidget(self.generate_button)
        generate_hlayout.addWidget(cancel_button)

        # Configuration controls
        self.config_frame = QFrame()
        self.config_frame.setContentsMargins(0, 0, 2, 0)

        config_scroll_area = ScrollArea()
        config_scroll_area.setFrameStyle(QFrame.NoFrame)
        config_scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        config_scroll_area.setWidgetResizable(True)
        config_scroll_area.setWidget(self.config_frame)
        config_scroll_area.setFocusPolicy(Qt.NoFocus)

        # Model
        self.model_combo_box = ComboBox()
        self.settings.beginGroup('Models')
        for key in self.settings.childKeys():
            value = self.settings.value(key)
            index = self.model_combo_box.addItem(key, value)
        self.settings.endGroup()
        utils.set_current_data(self.model_combo_box, self.settings.value('model'))

        # Prompts
        self.prompt_edit = PromptTextEdit(8, 'Prompt')
        self.prompt_edit.setPlainText(self.settings.value('prompt'))
        self.prompt_edit.return_pressed.connect(self.on_generate_image)
        self.negative_prompt_edit = PromptTextEdit(5, 'Negative Prompt')
        self.negative_prompt_edit.setPlainText(self.settings.value('negative_prompt'))
        self.negative_prompt_edit.return_pressed.connect(self.on_generate_image)

        self.prompt_group_box = QGroupBox('Prompts')
        prompt_group_box_layout = QVBoxLayout(self.prompt_group_box)
        prompt_group_box_layout.addWidget(self.prompt_edit)
        prompt_group_box_layout.addWidget(self.negative_prompt_edit)

        # General
        num_images_label = QLabel('Images')
        num_images_label.setAlignment(Qt.AlignCenter)
        self.num_images_spin_box = SpinBox()
        self.num_images_spin_box.setAlignment(Qt.AlignCenter)
        self.num_images_spin_box.setFixedWidth(80)
        self.num_images_spin_box.setMinimum(1)
        self.num_images_spin_box.setValue(int(self.settings.value('num_images_per_prompt')))
        num_steps_label = QLabel('Steps')
        num_steps_label.setAlignment(Qt.AlignCenter)
        self.num_steps_spin_box = SpinBox()
        self.num_steps_spin_box.setAlignment(Qt.AlignCenter)
        self.num_steps_spin_box.setFixedWidth(80)
        self.num_steps_spin_box.setMinimum(1)
        self.num_steps_spin_box.setValue(int(self.settings.value('num_inference_steps')))
        guidance_scale_label = QLabel('CFG Scale')
        guidance_scale_label.setAlignment(Qt.AlignCenter)
        self.guidance_scale_spin_box = DoubleSpinBox()
        self.guidance_scale_spin_box.setFocusPolicy(Qt.StrongFocus)
        self.guidance_scale_spin_box.setAlignment(Qt.AlignCenter)
        self.guidance_scale_spin_box.setFixedWidth(80)
        self.guidance_scale_spin_box.setSingleStep(0.5)
        self.guidance_scale_spin_box.setMinimum(1.0)
        self.guidance_scale_spin_box.setValue(float(self.settings.value('guidance_scale')))
        width_label = QLabel('Width')
        width_label.setAlignment(Qt.AlignCenter)
        self.width_spin_box = SpinBox()
        self.width_spin_box.setAlignment(Qt.AlignCenter)
        self.width_spin_box.setFixedWidth(80)
        self.width_spin_box.setSingleStep(64)
        self.width_spin_box.setMinimum(64)
        self.width_spin_box.setMaximum(1024)
        self.width_spin_box.setValue(int(self.settings.value('width')))
        height_label = QLabel('Height')
        height_label.setAlignment(Qt.AlignCenter)
        self.height_spin_box = SpinBox()
        self.height_spin_box.setAlignment(Qt.AlignCenter)
        self.height_spin_box.setFixedWidth(80)
        self.height_spin_box.setSingleStep(64)
        self.height_spin_box.setMinimum(64)
        self.height_spin_box.setMaximum(1024)
        self.height_spin_box.setValue(int(self.settings.value('height')))
        scheduler_label = QLabel('Scheduler')
        scheduler_label.setAlignment(Qt.AlignCenter)
        self.scheduler_combo_box = ComboBox()
        self.scheduler_combo_box.addItems(configuration.schedulers.keys())
        self.scheduler_combo_box.setFixedWidth(120)
        self.scheduler_combo_box.setCurrentText(self.settings.value('scheduler'))

        self.general_group_box = QGroupBox('General')
        controls_grid = QGridLayout(self.general_group_box)
        controls_grid.setVerticalSpacing(2)
        controls_grid.setRowMinimumHeight(2, 10)
        controls_grid.addWidget(num_images_label, 0, 0)
        controls_grid.addWidget(self.num_images_spin_box, 1, 0)
        controls_grid.addWidget(num_steps_label, 0, 1)
        controls_grid.addWidget(self.num_steps_spin_box, 1, 1)
        controls_grid.addWidget(guidance_scale_label, 0, 2)
        controls_grid.addWidget(self.guidance_scale_spin_box, 1, 2)
        controls_grid.setAlignment(self.guidance_scale_spin_box, Qt.AlignCenter)
        controls_grid.addWidget(width_label, 3, 0)
        controls_grid.addWidget(self.width_spin_box, 4, 0)
        controls_grid.addWidget(height_label, 3, 1)
        controls_grid.addWidget(self.height_spin_box, 4, 1)
        controls_grid.addWidget(scheduler_label, 3, 2)
        controls_grid.addWidget(self.scheduler_combo_box, 4, 2)

        # Seed
        self.seed_lineedit = QLineEdit()
        self.seed_lineedit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.seed_lineedit.setText(str(self.settings.value('seed')))
        seed_random_button = QPushButton('New')
        seed_random_button.clicked.connect(self.on_seed_random_clicked)

        self.seed_frame = QFrame()
        self.seed_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        seed_hlayout = QHBoxLayout(self.seed_frame)
        seed_hlayout.setContentsMargins(0, 0, 0, 0)
        seed_hlayout.addWidget(self.seed_lineedit)
        seed_hlayout.addWidget(seed_random_button)

        self.manual_seed_group_box = QGroupBox('Manual Seed')
        self.manual_seed_group_box.setCheckable(True)
        self.manual_seed_group_box.setChecked(self.settings.value('manual_seed', type=bool))
        manual_seed_group_box_layout = QVBoxLayout(self.manual_seed_group_box)
        manual_seed_group_box_layout.addWidget(self.seed_frame)

        # Source Images
        source_image_add_button = QPushButton('Add Image')
        source_image_add_button.clicked.connect(self.on_add_source_image)

        self.source_images_group_box = QGroupBox('Source Images')
        self.source_images_group_box_layout = QVBoxLayout(self.source_images_group_box)
        self.source_images_dynamic_index = self.source_images_group_box_layout.count()
        self.source_images_group_box_layout.addWidget(source_image_add_button)

        # Image to Image
        self.img2img_ref_ui = None
        self.img2img_ref_ui = self.create_image_ref_ui()
        self.img2img_strength = FloatSliderSpinBox('Strength', float(self.settings.value('img2img_strength')))

        self.img2img_group_box = QGroupBox('Image to Image')
        self.img2img_group_box.setCheckable(True)
        self.img2img_group_box.setChecked(self.settings.value('img2img_enabled', type=bool))
        self.img2img_group_box.toggled.connect(lambda: self.update_control_state())
        img2img_group_box_layout = QVBoxLayout(self.img2img_group_box)
        img2img_group_box_layout.addWidget(self.img2img_ref_ui.frame)
        img2img_group_box_layout.addWidget(self.img2img_strength)

        # ControlNet
        self.control_net_guidance_start = FloatSliderSpinBox('Guidance Start', float(self.settings.value('control_net_guidance_start')))
        self.control_net_guidance_end = FloatSliderSpinBox('Guidance End', float(self.settings.value('control_net_guidance_end')))

        self.control_net_add_button = QPushButton('Add Condition')
        self.control_net_add_button.clicked.connect(self.on_add_control_net)

        self.control_net_group_box = QGroupBox('Control Net')
        self.control_net_group_box.setCheckable(True)
        self.control_net_group_box.setChecked(self.settings.value('control_net_enabled', type=bool))
        self.control_net_group_box.toggled.connect(lambda: self.update_control_state())
        self.control_net_group_box_layout = QVBoxLayout(self.control_net_group_box)
        self.control_net_group_box_layout.addWidget(self.control_net_guidance_start)
        self.control_net_group_box_layout.addWidget(self.control_net_guidance_end)
        self.control_net_dynamic_index = self.control_net_group_box_layout.count()
        self.control_net_group_box_layout.addWidget(self.control_net_add_button)

        control_net_metas = [ControlNetMetadata.from_dict(item) for item in json.loads(settings.value('control_nets'))]
        for i, control_net_meta in enumerate(control_net_metas):
            control_net_frame = self.create_control_net_frame(control_net_meta)
            self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, control_net_frame)
            self.control_net_frames.append(control_net_frame)

        # Upscale
        upscale_factor_label = QLabel('Factor: ')
        self.upscale_factor_combo_box = ComboBox()
        self.upscale_factor_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.upscale_factor_combo_box.addItem('2x', 2)
        self.upscale_factor_combo_box.addItem('4x', 4)
        utils.set_current_data(self.upscale_factor_combo_box, self.settings.value('upscale_factor', type=int))
        self.upscale_denoising_strength = FloatSliderSpinBox('Denoising', float(self.settings.value('upscale_denoising_strength')))
        self.upscale_blend_strength = FloatSliderSpinBox('Strength', float(self.settings.value('upscale_blend_strength')))

        self.upscale_group_box = QGroupBox('Upscaling')
        self.upscale_group_box.setCheckable(True)
        self.upscale_group_box.setChecked(self.settings.value('upscale_enabled', type=bool))
        upscale_factor_layout = QHBoxLayout()
        upscale_factor_layout.setContentsMargins(0, 0, 0, 0) 
        upscale_factor_layout.setSpacing(0)
        upscale_factor_layout.addWidget(upscale_factor_label)
        upscale_factor_layout.addWidget(self.upscale_factor_combo_box)
        upscale_group_box_layout = QVBoxLayout(self.upscale_group_box)
        upscale_group_box_layout.addLayout(upscale_factor_layout)
        upscale_group_box_layout.addWidget(self.upscale_denoising_strength)
        upscale_group_box_layout.addWidget(self.upscale_blend_strength)

        # Face Restoration
        self.face_strength = FloatSliderSpinBox('Strength', float(self.settings.value('face_blend_strength')))

        self.face_strength_group_box = QGroupBox('Face Restoration')
        self.face_strength_group_box.setCheckable(True)
        self.face_strength_group_box.setChecked(self.settings.value('face_enabled', type=bool))

        face_strength_group_box_layout = QVBoxLayout(self.face_strength_group_box)
        face_strength_group_box_layout.addWidget(self.face_strength)

        # Configuration
        config_layout = QVBoxLayout(self.config_frame)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self.model_combo_box)
        config_layout.addWidget(self.prompt_group_box)
        config_layout.addWidget(self.source_images_group_box)
        config_layout.addWidget(self.general_group_box)
        config_layout.addWidget(self.manual_seed_group_box)
        config_layout.addWidget(self.img2img_group_box)
        config_layout.addWidget(self.control_net_group_box)
        config_layout.addWidget(self.upscale_group_box)
        config_layout.addWidget(self.face_strength_group_box)
        config_layout.addStretch()

        # Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.locate_source_button.pressed.connect(lambda: self.thumbnail_viewer.select_image(self.image_viewer.left_image_path()))
        self.image_viewer.send_to_img2img_button.pressed.connect(lambda: self.on_send_to_img2img(self.image_viewer.metadata))
        self.image_viewer.use_prompt_button.pressed.connect(lambda: self.on_use_prompt(self.image_viewer.metadata))
        self.image_viewer.use_seed_button.pressed.connect(lambda: self.on_use_seed(self.image_viewer.metadata))
        self.image_viewer.use_source_images_button.pressed.connect(lambda: self.on_use_source_images(self.image_viewer.metadata))
        self.image_viewer.use_all_button.pressed.connect(lambda: self.on_use_all(self.image_viewer.metadata))
        self.image_viewer.delete_button.pressed.connect(lambda: self.on_delete(self.image_viewer.metadata))

        image_viewer_frame = QFrame()
        image_viewer_frame.setFrameShape(QFrame.Panel)

        image_viewer_layout = QVBoxLayout(image_viewer_frame)
        image_viewer_layout.setContentsMargins(0, 0, 0, 0)
        image_viewer_layout.addWidget(self.image_viewer)

        # Thumbnail viewer
        self.thumbnail_viewer = ThumbnailViewer(self.settings, self.collections)
        self.thumbnail_viewer.file_dropped.connect(self.on_thumbnail_file_dropped)
        self.thumbnail_viewer.list_widget.itemSelectionChanged.connect(self.on_thumbnail_selection_change)
        self.thumbnail_viewer.action_send_to_img2img.triggered.connect(lambda: self.on_send_to_img2img(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_use_prompt.triggered.connect(lambda: self.on_use_prompt(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_use_seed.triggered.connect(lambda: self.on_use_seed(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_use_source_images.triggered.connect(lambda: self.on_use_source_images(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_use_all.triggered.connect(lambda: self.on_use_all(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_delete.triggered.connect(lambda: self.on_delete(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_reveal_in_finder.triggered.connect(lambda: self.on_reveal_in_finder(self.thumbnail_viewer.get_current_metadata()))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(image_viewer_frame)
        splitter.addWidget(self.thumbnail_viewer)
        splitter.setStretchFactor(0, 1)  # left widget
        splitter.setStretchFactor(1, 0)  # right widget

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setStyleSheet('QProgressBar { border: none; }')

        config_vlayout = QVBoxLayout()
        config_vlayout.addLayout(generate_hlayout)
        config_vlayout.addWidget(config_scroll_area)

        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(8, 2, 8, 8)
        hlayout.setSpacing(8)
        hlayout.addLayout(config_vlayout)
        hlayout.addWidget(splitter)

        vlayout = QVBoxLayout(central_widget)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        vlayout.addWidget(self.progress_bar)
        vlayout.addLayout(hlayout)

        self.setWindowTitle(configuration.APP_NAME)
        self.setGeometry(100, 100, 1200, 600)

        # Update state
        self.set_type(self.settings.value('type'))
        self.set_source_images(json.loads(self.settings.value('source_images')))
        self.update_image_ref_uis()
        self.on_thumbnail_selection_change()
        self.update_control_state()

    def create_source_image_ui(self) -> SourceImageUI:
        source_image_ui = SourceImageUI()

        source_image_ui.frame = QFrame();
        source_image_ui.frame.setContentsMargins(0, 0, 0, 0)

        source_image_ui.label = QLabel('X')
        source_image_ui.line_edit = QLineEdit()
        source_image_ui.line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_image_ui.line_edit.textChanged.connect(self.on_source_image_ui_text_changed)

        eye_button = QPushButton(fa.icon_eye)
        eye_button.setFont(fa.font)
        eye_button.setToolTip('View')
        eye_button.setToolTipDuration(0)
        eye_button.clicked.connect(lambda: self.set_current_source_image(source_image_ui))

        remove_button = QPushButton(fa.icon_xmark)
        remove_button.setFont(fa.font)
        remove_button.setToolTip('Remove')
        remove_button.setToolTipDuration(0)
        remove_button.clicked.connect(lambda: self.on_remove_source_image(source_image_ui))

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        button_layout.addWidget(eye_button)
        button_layout.addWidget(remove_button)

        hlayout = QHBoxLayout(source_image_ui.frame)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(2)
        hlayout.addWidget(source_image_ui.label)
        hlayout.addWidget(source_image_ui.line_edit)
        hlayout.addLayout(button_layout)

        return source_image_ui

    def create_image_ref_ui(self) -> ImageRefUI:
        image_ref_ui = ImageRefUI()

        image_ref_ui.frame = QFrame()
        image_ref_ui.frame.setContentsMargins(0, 0, 0, 0)

        image_ref_ui.combo_box = ComboBox()
        image_ref_ui.combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        eye_button = QPushButton(fa.icon_eye)
        eye_button.setFont(fa.font)
        eye_button.setToolTip('View')
        eye_button.setToolTipDuration(0)
        eye_button.clicked.connect(lambda: self.set_current_source_image(image_ref_ui.combo_box.currentData()))

        hlayout = QHBoxLayout(image_ref_ui.frame)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(2)
        hlayout.addWidget(image_ref_ui.combo_box)
        hlayout.addWidget(eye_button)

        return image_ref_ui

    def update_image_ref_uis(self):
        if self.img2img_ref_ui:
            self.update_image_ref_ui(self.img2img_ref_ui)
        for control_net_frame in self.control_net_frames:
            self.update_image_ref_ui(control_net_frame.image_ref_ui)

    def update_image_ref_ui(self, image_ref_ui: ImageRefUI):
        current_data = image_ref_ui.combo_box.currentData()
        image_ref_ui.combo_box.clear()

        for i, source_image_ui in enumerate(self.source_image_uis):
            text = '{:d} - {:s}'.format(i + 1, source_image_ui.line_edit.text())
            image_ref_ui.combo_box.addItem(text, source_image_ui)

        utils.set_current_data(image_ref_ui.combo_box, current_data)

    def create_control_net_frame(self, control_net_meta: ControlNetMetadata) -> ControlNetFrame:
        control_net_frame = ControlNetFrame()
        control_net_frame.setFrameStyle(QFrame.Box)

        control_net_frame.model_combo_box = ComboBox()
        control_net_frame.model_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_net_frame.model_combo_box.addItems(configuration.control_net_models.keys())
        control_net_frame.model_combo_box.setCurrentText(control_net_meta.name)

        remove_button = QPushButton()
        remove_button.setText(fa.icon_xmark)
        remove_button.setFont(fa.font)
        remove_button.setToolTip('Remove')
        remove_button.clicked.connect(lambda: self.on_remove_control_net(control_net_frame))

        control_net_frame.image_ref_ui = self.create_image_ref_ui()

        control_net_frame.preprocess_check_box = QCheckBox('Preprocess')
        control_net_frame.preprocess_check_box.setChecked(control_net_meta.preprocess)
        preview_preprocessor_button = QPushButton('Preview')
        preview_preprocessor_button.clicked.connect(lambda: self.on_control_net_preview_preprocessor_button_clicked(control_net_frame))

        control_net_frame.scale = FloatSliderSpinBox('Scale', control_net_meta.scale, maximum=2.0)

        model_hlayout = QHBoxLayout()
        model_hlayout.setContentsMargins(0, 0, 0, 0)
        model_hlayout.setSpacing(0)
        model_hlayout.addWidget(control_net_frame.model_combo_box)
        model_hlayout.addSpacing(2)
        model_hlayout.addWidget(remove_button)

        preprocess_hlayout = QHBoxLayout()
        preprocess_hlayout.setContentsMargins(0, 0, 0, 0)
        preprocess_hlayout.addWidget(control_net_frame.preprocess_check_box)
        preprocess_hlayout.addWidget(preview_preprocessor_button)

        control_net_layout = QVBoxLayout(control_net_frame)
        control_net_layout.addLayout(model_hlayout)
        control_net_layout.addWidget(control_net_frame.image_ref_ui.frame)
        control_net_layout.addLayout(preprocess_hlayout)
        control_net_layout.addWidget(control_net_frame.scale)
        return control_net_frame

    def show_about_dialog(self):
        about_dialog = AboutDialog()
        about_dialog.exec()

    def show_preferences_dialog(self):
        dialog = PreferencesDialog(self)
        dialog.exec()

    def set_type(self, type):
        self.type = type
        if self.type == 'image':
            self.button_group.button(0).setChecked(True)

    def set_current_source_image(self, source_image_ui: SourceImageUI):
        plain_font = QFont()
        bold_font = QFont()
        bold_font.setWeight(QFont.Bold)

        if self.current_source_image_ui:
            self.current_source_image_ui.line_edit.setFont(plain_font)

        self.current_source_image_ui = source_image_ui

        if source_image_ui:
            source_image_ui.line_edit.setFont(bold_font)
            self.image_viewer.set_left_image(source_image_ui.line_edit.text())
        else:
            self.image_viewer.clear_left_image()

    def on_mode_changed(self, button_id, checked):
        if not checked:
            return
        if button_id == 0:
            self.type = 'image'

    def on_cancel_generation(self):
        if self.generate_thread:
            self.generate_thread.cancel = True

    def set_source_images(self, source_images):
        if self.current_source_image_ui:
            index = self.source_images_group_box_layout.indexOf(self.current_source_image_ui.frame) - self.source_images_dynamic_index
        else:
            index = 0

        for source_image_ui in self.source_image_uis:
            self.source_images_group_box_layout.removeWidget(source_image_ui.frame)
            source_image_ui.frame.setParent(None)

        self.source_image_uis = []
        self.current_source_image_ui = None
        for i, source_path in enumerate(source_images):
            source_image_ui = self.create_source_image_ui()
            source_image_ui.label.setText(str(i+1))
            source_image_ui.line_edit.setText(source_path)
            self.source_images_group_box_layout.insertWidget(self.source_images_dynamic_index + i, source_image_ui.frame)
            self.source_image_uis.append(source_image_ui)

        self.update_control_state()
        self.update_image_ref_uis()
        self.config_frame.adjustSize()

        index = min(index, len(self.source_image_uis) - 1)
        self.set_current_source_image(self.source_image_uis[index] if index >= 0 else None)

    def on_add_source_image(self, path=None):
        i = len(self.source_image_uis)
        source_image_ui = self.create_source_image_ui()
        source_image_ui.label.setText(str(i+1))
        if path:
            source_image_ui.line_edit.setText(path)
        self.source_images_group_box_layout.insertWidget(self.source_images_dynamic_index + i, source_image_ui.frame)
        self.source_image_uis.append(source_image_ui)
        self.update_control_state()
        self.update_image_ref_uis()
        self.config_frame.adjustSize()

        self.set_current_source_image(source_image_ui)

    def on_remove_source_image(self, source_image_ui: SourceImageUI):
        index = self.source_images_group_box_layout.indexOf(source_image_ui.frame) - self.source_images_dynamic_index
        del self.source_image_uis[index]
        self.source_images_group_box_layout.removeWidget(source_image_ui.frame)
        source_image_ui.frame.setParent(None)
        for i, other_image_ui in enumerate(self.source_image_uis):
            other_image_ui.label.setText(str(i + 1))

        self.update_control_state()
        self.update_image_ref_uis()
        self.config_frame.adjustSize()

        if self.current_source_image_ui == source_image_ui:
            index = min(index, len(self.source_image_uis) - 1)
            self.set_current_source_image(self.source_image_uis[index] if index >= 0 else None)

    def on_add_control_net(self):
        control_net_meta = ControlNetMetadata()
        i = len(self.control_net_frames)
        control_net_frame = self.create_control_net_frame(control_net_meta)
        self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, control_net_frame)
        self.control_net_frames.append(control_net_frame)
        self.config_frame.adjustSize()

    def on_remove_control_net(self, widgetToRemove):
        index = self.control_net_group_box_layout.indexOf(widgetToRemove) - self.control_net_dynamic_index
        del self.control_net_frames[index]
        self.control_net_group_box_layout.removeWidget(widgetToRemove)
        widgetToRemove.setParent(None)
        self.config_frame.adjustSize()

    def on_source_image_ui_text_changed(self, text):
        if self.current_source_image_ui and self.current_source_image_ui.line_edit == self.sender():
            self.image_viewer.set_left_image(text)
        self.update_image_ref_uis()

    def on_control_net_preview_preprocessor_button_clicked(self, control_net_frame: ControlNetFrame):
        image_source = control_net_frame.image_ref_ui.combo_box.currentIndex()
        source_path = self.source_image_uis[image_source].line_edit.text()
        width = self.width_spin_box.value()
        height = self.height_spin_box.value()

        full_path = os.path.join(configuration.IMAGES_PATH, source_path)
        with Image.open(full_path) as image:
            image = image.convert('RGB')
            image = image.resize((width, height))
            source_image = image.copy()

        control_net_config = configuration.control_net_models[control_net_frame.model_combo_box.currentText()]
        if control_net_config:
            preprocessor_type = control_net_config.preprocessor
            if preprocessor_type:
                if not isinstance(self.preview_preprocessor, preprocessor_type):
                    self.preview_preprocessor = preprocessor_type()
                source_image = self.preview_preprocessor(source_image)
                if self.settings.value('reduce_memory', type=bool):
                    self.preview_preprocessor = None
                output_path = 'preprocessed.png'
                full_path = os.path.join(configuration.IMAGES_PATH, output_path)
                source_image.save(full_path)
                self.image_viewer.set_right_image(output_path)

    def update_control_state(self):
        self.image_viewer.set_both_images_visible(len(self.source_image_uis) > 0)

        # self.config_frame.adjustSize()

    def on_generate_image(self):
        if not self.manual_seed_group_box.isChecked():
            self.randomize_seed()

        control_net_metas = []
        for control_net_frame in self.control_net_frames:
            control_net_meta = ControlNetMetadata()
            control_net_meta.name = control_net_frame.model_combo_box.currentText()
            control_net_meta.image_source = control_net_frame.image_ref_ui.combo_box.currentIndex()
            control_net_meta.preprocess = control_net_frame.preprocess_check_box.isChecked()
            control_net_meta.scale = control_net_frame.scale.spin_box.value()
            control_net_metas.append(control_net_meta)

        if len(control_net_metas) == 0:
            control_net_meta = ControlNetMetadata()
            control_net_meta.name = 'Canny'
            control_net_metas.append(control_net_meta)

        source_image_paths = []
        for source_image_ui in self.source_image_uis:
            source_image_paths.append(source_image_ui.line_edit.text())

        self.settings.setValue('collection', self.thumbnail_viewer.collection())
        self.settings.setValue('type', self.type)
        self.settings.setValue('model', self.model_combo_box.currentData())
        self.settings.setValue('scheduler', self.scheduler_combo_box.currentText())
        self.settings.setValue('prompt', self.prompt_edit.toPlainText())
        self.settings.setValue('negative_prompt', self.negative_prompt_edit.toPlainText())
        self.settings.setValue('source_images', json.dumps(source_image_paths))
        self.settings.setValue('manual_seed', self.manual_seed_group_box.isChecked())
        self.settings.setValue('seed', self.seed_lineedit.text())
        self.settings.setValue('num_images_per_prompt', self.num_images_spin_box.value())
        self.settings.setValue('num_inference_steps', self.num_steps_spin_box.value())
        self.settings.setValue('guidance_scale', self.guidance_scale_spin_box.value())
        self.settings.setValue('width', self.width_spin_box.value())
        self.settings.setValue('height', self.height_spin_box.value())
        self.settings.setValue('img2img_enabled', self.img2img_group_box.isChecked())
        self.settings.setValue('img2img_source', self.img2img_ref_ui.combo_box.currentIndex())
        self.settings.setValue('img2img_strength', self.img2img_strength.spin_box.value())
        self.settings.setValue('control_net_enabled', self.control_net_group_box.isChecked())
        self.settings.setValue('control_net_guidance_start', self.control_net_guidance_start.spin_box.value())
        self.settings.setValue('control_net_guidance_end', self.control_net_guidance_end.spin_box.value())
        self.settings.setValue('control_nets', json.dumps([control_net.to_dict() for control_net in control_net_metas]))
        self.settings.setValue('upscale_enabled', self.upscale_group_box.isChecked())
        self.settings.setValue('upscale_factor', self.upscale_factor_combo_box.currentData())
        self.settings.setValue('upscale_denoising_strength', self.upscale_denoising_strength.spin_box.value())
        self.settings.setValue('upscale_blend_strength', self.upscale_blend_strength.spin_box.value())
        self.settings.setValue('face_enabled', self.face_strength_group_box.isChecked())
        self.settings.setValue('face_blend_strength', self.face_strength.spin_box.value())

        self.update_progress(0, 0)
        self.generate_button.setEnabled(False)
        self.generate_thread = GenerateThread(self.settings, self)
        self.generate_thread.task_progress.connect(self.update_progress)
        self.generate_thread.image_preview.connect(self.image_preview)
        self.generate_thread.image_complete.connect(self.image_complete)
        self.generate_thread.task_complete.connect(self.generate_complete)
        self.generate_thread.start()

    def update_progress(self, progress_amount, maximum_amount=100):
        self.progress_bar.setMaximum(maximum_amount)
        if maximum_amount == 0:
            self.progress_bar.setStyleSheet('QProgressBar { border: none; } QProgressBar:chunk { background-color: grey; }')
        else:
            self.progress_bar.setStyleSheet('QProgressBar { border: none; } QProgressBar:chunk { background-color: blue; }')
        if progress_amount is not None:
            self.progress_bar.setValue(progress_amount)
        else:
            self.progress_bar.setValue(0)

        if sys.platform == 'darwin':
            sharedApplication = NSApplication.sharedApplication()
            dockTile = sharedApplication.dockTile()
            if maximum_amount == 0:
                dockTile.setBadgeLabel_('...')
            elif progress_amount is not None:
                dockTile.setBadgeLabel_('{:d}%'.format(progress_amount))
            else:
                dockTile.setBadgeLabel_(None)

    def image_preview(self, preview_image):
        self.image_viewer.set_preview_image(preview_image)

    def image_complete(self, output_path):
        self.on_add_file(output_path)

    def generate_complete(self):
        self.generate_button.setEnabled(True)
        self.update_progress(None)
        self.image_viewer.set_preview_image(None)
        self.generate_thread = None

    def randomize_seed(self):
        seed = random.randint(0, 0x7fff_ffff_ffff_ffff)
        self.seed_lineedit.setText(str(seed))

    def on_seed_random_clicked(self):
        self.randomize_seed()

    def on_thumbnail_file_dropped(self, source_path: str):
        collection = self.thumbnail_viewer.collection()

        def io_operation():
            next_image_id = utils.next_image_id(os.path.join(configuration.IMAGES_PATH, collection))
            output_path = os.path.join(collection, '{:05d}.png'.format(next_image_id))
            full_path = os.path.join(configuration.IMAGES_PATH, output_path)

            with Image.open(source_path) as image:
                metadata = ImageMetadata()
                metadata.path = output_path
                metadata.load_from_image_info(image.info)
                png_info = PngImagePlugin.PngInfo()
                metadata.save_to_png_info(png_info)

                image.save(full_path, pnginfo = png_info)
            return output_path

        output_path = utils.retry_on_failure(io_operation)

        self.on_add_file(output_path)

    def on_thumbnail_selection_change(self):
        selected_items = self.thumbnail_viewer.list_widget.selectedItems()
        for item in selected_items:
            rel_path = item.data(Qt.UserRole)
            self.image_viewer.set_right_image(rel_path)

    def on_send_to_img2img(self, image_metadata):
        if image_metadata is not None:
            path = image_metadata.path
            if self.current_source_image_ui is None:
                self.on_add_source_image(path)
            else:
                self.current_source_image_ui.line_edit.setText(path)
    
    def on_use_prompt(self, image_metadata):
        if image_metadata is not None:
            self.prompt_edit.setPlainText(image_metadata.prompt)
            self.negative_prompt_edit.setPlainText(image_metadata.negative_prompt)

    def on_use_seed(self, image_metadata):
        if image_metadata is not None:
            self.manual_seed_group_box.setChecked(True)
            self.seed_lineedit.setText(str(image_metadata.seed))

    def on_use_general(self, image_metadata):
        if image_metadata is not None:
            self.num_steps_spin_box.setValue(image_metadata.num_inference_steps)
            self.guidance_scale_spin_box.setValue(image_metadata.guidance_scale)
            self.width_spin_box.setValue(image_metadata.width)
            self.height_spin_box.setValue(image_metadata.height)
            self.scheduler_combo_box.setCurrentText(image_metadata.scheduler)

    def on_use_source_images(self, image_metadata):
        if image_metadata is not None:
            self.set_source_images(image_metadata.source_images)

    def on_use_img2img(self, image_metadata):
        if image_metadata is not None:
            if image_metadata.img2img_enabled:
                self.img2img_group_box.setChecked(True)
                self.img2img_ref_ui.combo_box.setCurrentIndex(image_metadata.img2img_source)
                self.img2img_strength.spin_box.setValue(image_metadata.img2img_strength)
            else:
                self.img2img_group_box.setChecked(False)

    def on_use_controlnet(self, image_metadata):
        if image_metadata is not None:
            if image_metadata.control_net_enabled:
                self.control_net_group_box.setChecked(True)
                self.control_net_guidance_start.spin_box.setValue(image_metadata.control_net_guidance_start)
                self.control_net_guidance_end.spin_box.setValue(image_metadata.control_net_guidance_end)

                for control_net_frame in self.control_net_frames:
                    self.control_net_group_box_layout.removeWidget(control_net_frame)
                    control_net_frame.setParent(None)

                self.control_net_frames = []
                for i, control_net_meta in enumerate(image_metadata.control_nets):
                    control_net_frame = self.create_control_net_frame(control_net_meta)
                    self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, control_net_frame)
                    self.control_net_frames.append(control_net_frame)
            else:
                self.control_net_group_box.setChecked(False)

    def on_use_post_processing(self, image_metadata):
        if image_metadata is not None:
            if image_metadata.upscale_enabled:
                self.upscale_group_box.setChecked(True)
                utils.set_current_data(self.upscale_factor_combo_box, image_metadata.upscale_factor)
                self.upscale_denoising_strength.spin_box.setValue(image_metadata.upscale_denoising_strength)
                self.upscale_blend_strength.spin_box.setValue(image_metadata.upscale_blend_strength)
            else:
                self.upscale_group_box.setChecked(False)

            if image_metadata.face_enabled:
                self.face_strength_group_box.setChecked(True)
                self.face_strength.spin_box.setValue(image_metadata.face_blend_strength)
            else:
                self.face_strength_group_box.setChecked(False)
 
    def on_use_all(self, image_metadata):
        if image_metadata is not None:
            self.on_use_prompt(image_metadata)
            self.on_use_seed(image_metadata)
            self.on_use_general(image_metadata)
            self.on_use_source_images(image_metadata)
            self.on_use_img2img(image_metadata)
            self.on_use_controlnet(image_metadata)
            self.on_use_post_processing(image_metadata)

    def on_move_image(self, image_metadata: ImageMetadata, collection: str):
        current_collection = self.thumbnail_viewer.collection()
        if collection == current_collection:
            return

        if image_metadata is not None:
            source_path = image_metadata.path

            def io_operation():
                next_image_id = utils.next_image_id(os.path.join(configuration.IMAGES_PATH, collection))
                output_path = os.path.join(collection, '{:05d}.png'.format(next_image_id))

                full_source_path = os.path.join(configuration.IMAGES_PATH, source_path)
                full_output_path = os.path.join(configuration.IMAGES_PATH, output_path)

                shutil.move(full_source_path, full_output_path)
                return output_path

            output_path = utils.retry_on_failure(io_operation)

            self.on_remove_file(image_metadata.path)
            self.on_add_file(output_path)

    def on_delete(self, image_metadata):
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

    def on_add_file(self, path):
        collection = os.path.dirname(path)
        current_collection = self.thumbnail_viewer.collection()
        if collection != current_collection:
            return

        self.thumbnail_viewer.add_image(path)
        self.thumbnail_viewer.select_index_no_scroll(0)
        self.image_viewer.set_right_image(path)

    def on_remove_file(self, path):
        full_thumbnail_path = os.path.join(configuration.THUMBNAILS_PATH, path)
        os.remove(full_thumbnail_path)

        self.thumbnail_viewer.remove_image(path)
        if self.image_viewer.left_image_path() == path:
            self.image_viewer.clear_left_image()

    def on_reveal_in_finder(self, image_metadata):
        if image_metadata is not None:
            full_path = os.path.abspath(os.path.join(configuration.IMAGES_PATH, image_metadata.path))
            utils.reveal_in_finder(full_path)

    def hide_if_thread_running(self):
        if self.generate_thread:
            self.active_thread_count = self.active_thread_count + 1
            self.generate_thread.cancel = True
            self.generate_thread.finished.connect(self.thread_finished)

        if self.active_thread_count > 0:
            self.hide()
            return True
        else:
            return False

    def thread_finished(self):
        self.active_thread_count = self.active_thread_count - 1
        if self.active_thread_count == 0:
            QApplication.instance().quit()

    def closeEvent(self, event):
        if self.hide_if_thread_running():
            event.ignore()
        else:
            event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self.thumbnail_viewer.previous_image()
        elif key == Qt.Key_Right:
            self.thumbnail_viewer.next_image()
        else:
            super().keyPressEvent(event)
