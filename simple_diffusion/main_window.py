import json
import os
import random
import shutil
import sys

from PIL import Image, PngImagePlugin
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDialog,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMenu, QMenuBar,
                               QProgressBar, QPushButton, QSizePolicy,
                               QSplitter, QToolBar, QVBoxLayout, QWidget)

from . import actions, configuration
from . import font_awesome as fa
from . import utils
from .about_dialog import AboutDialog
from .delete_image_dialog import DeleteImageDialog
from .generate_thread import GenerateThread
from .image_history import ImageHistory
from .image_metadata import ControlNetMetadata, ImageMetadata
from .image_viewer import ImageViewer
from .preferences_dialog import PreferencesDialog
from .processors import ProcessorBase
from .prompt_text_edit import PromptTextEdit
from .thumbnail_loader import ThumbnailLoader
from .thumbnail_viewer import ThumbnailViewer
from .widgets import (ComboBox, DoubleSpinBox, FloatSliderSpinBox,
                      IntSliderSpinBox, ScrollArea, SpinBox)

if sys.platform == 'darwin':
    from AppKit import NSApplication

class SourceImageUI:
    frame: QFrame
    label: QLabel
    line_edit: QLineEdit
    context_menu: QMenu

class ControlNetFrame(QFrame):
    model_combo_box: ComboBox = None
    source_image_ui: SourceImageUI = None
    preprocess_check_box: QCheckBox = None
    scale: FloatSliderSpinBox = None

class MainWindow(QMainWindow):
    preview_preprocessor: ProcessorBase = None
    img2img_source_ui: SourceImageUI = None
    control_net_frames: list[ControlNetFrame] = []
    override_metadata = None

    def __init__(self, settings: QSettings, collections: list[str]):
        super().__init__()

        self.settings = settings
        self.collections = collections
        self.image_history = ImageHistory()
        self.image_history.current_image_changed.connect(self.on_current_image_changed)
        self.thumbnail_loader = ThumbnailLoader()

        self.generate_thread = None
        self.active_thread_count = 0

        self.setFocusPolicy(Qt.ClickFocus)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Set as Source Menu
        self.set_as_source_menu = QMenu('Set as Source', self)
        self.set_as_source_menu.setIcon(QIcon(utils.create_fontawesome_icon(fa.icon_share)))
        self.set_as_source_menu.addAction(QAction("Dummy...", self))
        self.set_as_source_menu.aboutToShow.connect(self.populate_set_as_source_menu)

        # Move To Menu
        move_image_menu = QMenu('Move To', self)
        move_image_menu.setIcon(utils.empty_qicon())
        for collection in self.collections:
            item_action = QAction(collection, self)
            item_action.triggered.connect(lambda checked=False, collection=collection: self.on_move_image(collection))
            move_image_menu.addAction(item_action)

        # File Menu
        action_preferences = actions.preferences.create(self)
        action_preferences.triggered.connect(self.show_preferences_dialog)
        action_exit = actions.exit.create(self)
        action_exit.triggered.connect(self.close)

        app_menu = QMenu("File", self)
        app_menu.addAction(action_preferences)
        app_menu.addSeparator()
        app_menu.addAction(action_exit)

        # History Menu
        history_menu = QMenu("History", self)
        history_menu.addAction(QAction("Dummy...", self))
        history_menu.aboutToShow.connect(lambda: self.image_history.populate_history_menu(history_menu))

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
        image_menu.addMenu(self.set_as_source_menu)
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

        # Help Menu
        action_about = actions.about.create(self)
        action_about.triggered.connect(self.show_about_dialog)

        help_menu = QMenu('Help', self)
        help_menu.addAction(action_about)

        # Menu bar
        menu_bar = QMenuBar(self)
        menu_bar.addMenu(app_menu)
        menu_bar.addMenu(history_menu)
        menu_bar.addMenu(image_menu)
        menu_bar.addMenu(help_menu)
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
        for model in configuration.known_stable_diffusion_models:
            self.model_combo_box.addItem(model, configuration.get_stable_diffusion_model_path(model))
        for repo_id in utils.deserialize_string_list(self.settings.value('huggingface_models')):
            self.model_combo_box.addItem(os.path.basename(repo_id), repo_id)
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
        self.width_spin_box.setMaximum(2048)
        self.width_spin_box.setValue(int(self.settings.value('width')))
        height_label = QLabel('Height')
        height_label.setAlignment(Qt.AlignCenter)
        self.height_spin_box = SpinBox()
        self.height_spin_box.setAlignment(Qt.AlignCenter)
        self.height_spin_box.setFixedWidth(80)
        self.height_spin_box.setSingleStep(64)
        self.height_spin_box.setMinimum(64)
        self.height_spin_box.setMaximum(2048)
        self.height_spin_box.setValue(int(self.settings.value('height')))
        scheduler_label = QLabel('Scheduler')
        scheduler_label.setAlignment(Qt.AlignCenter)
        self.scheduler_combo_box = ComboBox()
        self.scheduler_combo_box.addItems(configuration.schedulers.keys())
        self.scheduler_combo_box.setFixedWidth(120)
        self.scheduler_combo_box.setCurrentText(self.settings.value('scheduler'))

        self.general_group_box = QGroupBox('General')
        controls_grid = QGridLayout()
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

        controls_vlayout = QVBoxLayout(self.general_group_box)
        controls_vlayout.addWidget(self.model_combo_box)
        controls_vlayout.addLayout(controls_grid)

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

        # Image to Image
        self.img2img_source_ui = self.create_source_image_ui(self.settings.value('img2img_source', type=str))
        self.img2img_strength = FloatSliderSpinBox('Strength', float(self.settings.value('img2img_strength')))

        self.img2img_group_box = QGroupBox('Image to Image')
        self.img2img_group_box.setCheckable(True)
        self.img2img_group_box.setChecked(self.settings.value('img2img_enabled', type=bool))
        img2img_group_box_layout = QVBoxLayout(self.img2img_group_box)
        img2img_group_box_layout.addWidget(self.img2img_source_ui.frame)
        img2img_group_box_layout.addWidget(self.img2img_strength)

        # ControlNet
        self.control_net_guidance_start = FloatSliderSpinBox('Guidance Start', float(self.settings.value('control_net_guidance_start')))
        self.control_net_guidance_end = FloatSliderSpinBox('Guidance End', float(self.settings.value('control_net_guidance_end')))

        self.control_net_add_button = QPushButton('Add Condition')
        self.control_net_add_button.clicked.connect(self.on_add_control_net)

        self.control_net_group_box = QGroupBox('Control Net')
        self.control_net_group_box.setCheckable(True)
        self.control_net_group_box.setChecked(self.settings.value('control_net_enabled', type=bool))
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

        # High Resolution
        self.high_res_factor = FloatSliderSpinBox('Factor', self.settings.value('high_res_factor', type=float), minimum=1, maximum=2)
        self.high_res_steps = IntSliderSpinBox('Steps', self.settings.value('high_res_steps', type=int), minimum=1, maximum=200)
        self.high_res_guidance_scale = FloatSliderSpinBox('Guidance', self.settings.value('high_res_guidance_scale', type=float), minimum=1, maximum=50, single_step=0.5)
        self.high_res_noise = FloatSliderSpinBox('Noise', self.settings.value('high_res_noise', type=float))

        self.high_res_group_box = QGroupBox('High Resolution')
        self.high_res_group_box.setCheckable(True)
        self.high_res_group_box.setChecked(self.settings.value('high_res_enabled', type=bool))

        high_res_group_box_layout = QVBoxLayout(self.high_res_group_box)
        high_res_group_box_layout.addWidget(self.high_res_factor)
        high_res_group_box_layout.addWidget(self.high_res_steps)
        high_res_group_box_layout.addWidget(self.high_res_guidance_scale)
        high_res_group_box_layout.addWidget(self.high_res_noise)

        # Configuration
        config_layout = QVBoxLayout(self.config_frame)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self.prompt_group_box)
        config_layout.addWidget(self.general_group_box)
        config_layout.addWidget(self.manual_seed_group_box)
        config_layout.addWidget(self.img2img_group_box)
        config_layout.addWidget(self.control_net_group_box)
        config_layout.addWidget(self.upscale_group_box)
        config_layout.addWidget(self.face_strength_group_box)
        config_layout.addWidget(self.high_res_group_box)
        config_layout.addStretch()

        # Image viewer
        self.image_viewer = ImageViewer(self.image_history)
        self.image_viewer.set_as_source_image_button.setMenu(self.set_as_source_menu)
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
        thumbnail_menu.addMenu(self.set_as_source_menu)
        thumbnail_menu.addSeparator()
        thumbnail_menu.addAction(thumbnail_use_prompt_action)
        thumbnail_menu.addAction(thumbnail_use_seed_action)
        thumbnail_menu.addAction(thumbnail_use_source_images_action)
        thumbnail_menu.addAction(thumbnail_use_all_action)
        thumbnail_menu.addSeparator()
        thumbnail_menu.addMenu(move_image_menu)
        thumbnail_menu.addAction(thumbnail_delete_action)
        thumbnail_menu.addSeparator()
        thumbnail_menu.addAction(thumbnail_reveal_in_finder_action)

        self.thumbnail_viewer = ThumbnailViewer(self.thumbnail_loader, self.settings, self.collections, thumbnail_menu)
        self.thumbnail_viewer.file_dropped.connect(self.on_thumbnail_file_dropped)
        self.thumbnail_viewer.list_widget.image_selection_changed.connect(self.on_thumbnail_selection_change)

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
        selected_image = self.thumbnail_viewer.list_widget.selected_image()
        if selected_image is not None:
            self.image_history.visit(selected_image)

    def get_current_metadata(self):
        if self.override_metadata is not None:
            return self.override_metadata
        else:
            return self.image_viewer.metadata

    def set_override_metadata(self, metadata):
        self.override_metadata = metadata

    def create_source_image_ui(self, text: str) -> SourceImageUI:
        source_image_ui = SourceImageUI()

        source_image_ui.frame = QFrame()
        source_image_ui.frame.setContentsMargins(0, 0, 0, 0)

        source_image_ui.label = QLabel()
        source_image_ui.label.setContextMenuPolicy(Qt.CustomContextMenu)
        source_image_ui.label.customContextMenuRequested.connect(lambda point: self.show_source_image_context_menu(source_image_ui, point))

        source_image_ui.line_edit = QLineEdit()
        source_image_ui.line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_image_ui.line_edit.textChanged.connect(lambda: self.on_source_image_ui_text_changed(source_image_ui))
        source_image_ui.line_edit.setText(text)

        locate_source_action = actions.locate_source.create(self)
        locate_source_action.triggered.connect(lambda: self.thumbnail_viewer.select_image(source_image_ui.line_edit.text()))

        source_image_ui.context_menu = QMenu(self)
        source_image_ui.context_menu.aboutToShow.connect(lambda: self.set_override_metadata(self.get_source_image_metadata(source_image_ui)))
        source_image_ui.context_menu.aboutToHide.connect(lambda: QTimer.singleShot(0, lambda: self.set_override_metadata(None)))
        source_image_ui.context_menu.addAction(locate_source_action)
        source_image_ui.context_menu.addSeparator()
        source_image_ui.context_menu.addMenu(self.set_as_source_menu)

        hlayout = QHBoxLayout(source_image_ui.frame)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(2)
        hlayout.addWidget(source_image_ui.line_edit)
        hlayout.addWidget(source_image_ui.label)

        return source_image_ui
        
    def on_source_image_ui_text_changed(self, source_image_ui: SourceImageUI):
        image_path = source_image_ui.line_edit.text()
        self.thumbnail_loader.get(image_path, 96, lambda image_path, pixmap: self.on_thumbnail_loaded(source_image_ui, pixmap))

    def get_source_image_metadata(self, source_image_ui: SourceImageUI):
        image_path = source_image_ui.line_edit.text()
        if image_path:
            full_path = os.path.join(configuration.IMAGES_PATH, image_path)
            with Image.open(full_path) as image:
                metadata = ImageMetadata()
                metadata.path = image_path
                metadata.load_from_image(image)
                return metadata
        return None

    def show_source_image_context_menu(self, source_image_ui: SourceImageUI, point):
        source_image_ui.context_menu.exec(source_image_ui.label.mapToGlobal(point))

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

        control_net_frame.source_image_ui = self.create_source_image_ui(control_net_meta.image_source)

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
        control_net_layout.addWidget(control_net_frame.source_image_ui.frame)
        control_net_layout.addLayout(preprocess_hlayout)
        control_net_layout.addWidget(control_net_frame.scale)
        return control_net_frame
    
    def populate_set_as_source_menu(self):
        self.set_as_source_menu.clear()
        action = QAction('Image to Image', self)
        action.triggered.connect(lambda: self.on_set_as_source(self.img2img_source_ui))
        self.set_as_source_menu.addAction(action)

        for i, control_net_frame in enumerate(self.control_net_frames):
            action = QAction('Control Net {:d}'.format(i + 1), self)
            action.triggered.connect(lambda checked=False, control_net_frame=control_net_frame: self.on_set_as_source(control_net_frame.source_image_ui))
            self.set_as_source_menu.addAction(action)
        
        self.set_as_source_menu.addSeparator()

        action = QAction('New Control Net Condition', self)
        action.triggered.connect(lambda: self.on_set_as_source(self.on_add_control_net().source_image_ui))
        self.set_as_source_menu.addAction(action)
    
    def on_set_as_source(self, source_image_ui: SourceImageUI):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            source_image_ui.line_edit.setText(image_metadata.path)
            if source_image_ui == self.img2img_source_ui:
                self.img2img_group_box.setChecked(True)
                self.width_spin_box.setValue(image_metadata.width)
                self.height_spin_box.setValue(image_metadata.height)
            else:
                self.control_net_group_box.setChecked(True)
    
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

    def on_mode_changed(self, button_id, checked):
        if not checked:
            return
        if button_id == 0:
            self.type = 'image'

    def on_cancel_generation(self):
        if self.generate_thread:
            self.generate_thread.cancel = True

    def on_add_control_net(self):
        control_net_meta = ControlNetMetadata()
        i = len(self.control_net_frames)
        control_net_frame = self.create_control_net_frame(control_net_meta)
        self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, control_net_frame)
        self.control_net_frames.append(control_net_frame)
        self.config_frame.adjustSize()
        return control_net_frame

    def on_remove_control_net(self, widgetToRemove):
        index = self.control_net_group_box_layout.indexOf(widgetToRemove) - self.control_net_dynamic_index
        del self.control_net_frames[index]
        self.control_net_group_box_layout.removeWidget(widgetToRemove)
        widgetToRemove.setParent(None)
        self.config_frame.adjustSize()

    def on_thumbnail_loaded(self, source_image_ui, pixmap):
        source_image_ui.label.setPixmap(pixmap)

    def on_control_net_preview_preprocessor_button_clicked(self, control_net_frame: ControlNetFrame):
        source_path = control_net_frame.source_image_ui.line_edit.text()
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
                self.image_viewer.set_current_image(output_path)

    def on_generate_image(self):
        if not self.manual_seed_group_box.isChecked():
            self.randomize_seed()

        control_net_metas = []
        for control_net_frame in self.control_net_frames:
            control_net_meta = ControlNetMetadata()
            control_net_meta.name = control_net_frame.model_combo_box.currentText()
            control_net_meta.image_source = control_net_frame.source_image_ui.line_edit.text()
            control_net_meta.preprocess = control_net_frame.preprocess_check_box.isChecked()
            control_net_meta.scale = control_net_frame.scale.spin_box.value()
            control_net_metas.append(control_net_meta)

        self.settings.setValue('collection', self.thumbnail_viewer.collection())
        self.settings.setValue('type', self.type)
        self.settings.setValue('model', self.model_combo_box.currentData())
        self.settings.setValue('scheduler', self.scheduler_combo_box.currentText())
        self.settings.setValue('prompt', self.prompt_edit.toPlainText())
        self.settings.setValue('negative_prompt', self.negative_prompt_edit.toPlainText())
        self.settings.setValue('manual_seed', self.manual_seed_group_box.isChecked())
        self.settings.setValue('seed', self.seed_lineedit.text())
        self.settings.setValue('num_images_per_prompt', self.num_images_spin_box.value())
        self.settings.setValue('num_inference_steps', self.num_steps_spin_box.value())
        self.settings.setValue('guidance_scale', self.guidance_scale_spin_box.value())
        self.settings.setValue('width', self.width_spin_box.value())
        self.settings.setValue('height', self.height_spin_box.value())
        self.settings.setValue('img2img_enabled', self.img2img_group_box.isChecked())
        self.settings.setValue('img2img_source', self.img2img_source_ui.line_edit.text())
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
        self.settings.setValue('high_res_enabled', self.high_res_group_box.isChecked())
        self.settings.setValue('high_res_factor', self.high_res_factor.spin_box.value())
        self.settings.setValue('high_res_steps', self.high_res_steps.spin_box.value())
        self.settings.setValue('high_res_guidance_scale', self.high_res_guidance_scale.spin_box.value())
        self.settings.setValue('high_res_noise', self.high_res_noise.spin_box.value())

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

    def on_current_image_changed(self, path):
        self.image_viewer.set_current_image(path)
        self.thumbnail_viewer.list_widget.image_selection_changed.disconnect()
        self.thumbnail_viewer.select_image(path)
        self.thumbnail_viewer.list_widget.image_selection_changed.connect(self.on_thumbnail_selection_change)

    def on_thumbnail_file_dropped(self, source_path: str):
        collection = self.thumbnail_viewer.collection()

        def io_operation():
            next_image_id = utils.next_image_id(os.path.join(configuration.IMAGES_PATH, collection))
            output_path = os.path.join(collection, '{:05d}.png'.format(next_image_id))
            full_path = os.path.join(configuration.IMAGES_PATH, output_path)

            with Image.open(source_path) as image:
                metadata = ImageMetadata()
                metadata.path = output_path
                metadata.load_from_image(image)
                png_info = PngImagePlugin.PngInfo()
                metadata.save_to_png_info(png_info)

                image.save(full_path, pnginfo = png_info)
            return output_path

        output_path = utils.retry_on_failure(io_operation)

        self.on_add_file(output_path)

    def on_thumbnail_selection_change(self, image_path):
        self.image_history.visit(image_path)

    def on_use_prompt(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.prompt_edit.setPlainText(image_metadata.prompt)
            self.negative_prompt_edit.setPlainText(image_metadata.negative_prompt)

    def on_use_seed(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.manual_seed_group_box.setChecked(True)
            self.seed_lineedit.setText(str(image_metadata.seed))

    def on_use_general(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.num_steps_spin_box.setValue(image_metadata.num_inference_steps)
            self.guidance_scale_spin_box.setValue(image_metadata.guidance_scale)
            self.width_spin_box.setValue(image_metadata.width)
            self.height_spin_box.setValue(image_metadata.height)
            self.scheduler_combo_box.setCurrentText(image_metadata.scheduler)

    def on_use_source_images(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.img2img_source_ui.line_edit.setText(image_metadata.img2img_source)

            while len(self.control_net_frames) < len(image_metadata.control_nets):
                self.on_add_control_net()
            
            for i, control_net_meta in enumerate(image_metadata.control_nets):
                control_net_frame = self.control_net_frames[i]
                control_net_frame.source_image_ui.line_edit.setText(control_net_meta.image_source)

    def on_use_img2img(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            if image_metadata.img2img_enabled:
                self.img2img_group_box.setChecked(True)
                self.img2img_source_ui.line_edit.setText(image_metadata.img2img_source)
                self.img2img_strength.spin_box.setValue(image_metadata.img2img_strength)
            else:
                self.img2img_group_box.setChecked(False)

    def on_use_controlnet(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            for control_net_frame in self.control_net_frames:
                self.control_net_group_box_layout.removeWidget(control_net_frame)
                control_net_frame.setParent(None)

            self.control_net_frames = []

            if image_metadata.control_net_enabled:
                self.control_net_group_box.setChecked(True)
                self.control_net_guidance_start.spin_box.setValue(image_metadata.control_net_guidance_start)
                self.control_net_guidance_end.spin_box.setValue(image_metadata.control_net_guidance_end)

                for i, control_net_meta in enumerate(image_metadata.control_nets):
                    control_net_frame = self.create_control_net_frame(control_net_meta)
                    self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, control_net_frame)
                    self.control_net_frames.append(control_net_frame)
            else:
                self.control_net_group_box.setChecked(False)

    def on_use_post_processing(self):
        image_metadata = self.get_current_metadata()
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
            
            if image_metadata.high_res_enabled:
                self.high_res_group_box.setChecked(True)
                self.high_res_factor.spin_box.setValue(image_metadata.high_res_factor)
                self.high_res_steps.spin_box.setValue(image_metadata.high_res_steps)
                self.high_res_guidance_scale.spin_box.setValue(image_metadata.high_res_guidance_scale)
                self.high_res_noise.spin_box.setValue(image_metadata.high_res_noise)
            else:
                self.high_res_group_box.setChecked(False)
 
    def on_use_all(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.on_use_prompt()
            self.on_use_seed()
            self.on_use_general()
            self.on_use_img2img()
            self.on_use_controlnet()
            self.on_use_post_processing()
            self.on_use_source_images()

    def on_move_image(self, collection: str):
        current_collection = self.thumbnail_viewer.collection()
        if collection == current_collection:
            return

        image_metadata = self.get_current_metadata()
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

    def on_delete(self):
        image_metadata = self.get_current_metadata()
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
        image_metadata = self.get_current_metadata()
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
        self.thumbnail_loader.shutdown()
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
