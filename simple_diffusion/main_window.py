import os
import random
import shutil
import sys

import actions
import configuration
import utils
from about_dialog import AboutDialog
from configuration import ControlNetCondition, Img2ImgCondition
from delete_image_dialog import DeleteImageDialog
from float_slider_spin_box import FloatSliderSpinBox
from generate_thread import GenerateThread
from image_metadata import ImageMetadata
from image_viewer import ImageViewer
from PIL import Image, PngImagePlugin
from preferences_dialog import PreferencesDialog
from processors import ProcessorBase
from prompt_text_edit import PromptTextEdit
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox,
                               QComboBox, QDialog, QDoubleSpinBox, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMenu, QMenuBar, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSpinBox,
                               QSplitter, QToolBar, QVBoxLayout, QWidget)
from thumbnail_viewer import ThumbnailViewer

if sys.platform == 'darwin':
    from AppKit import NSApplication

class MainWindow(QMainWindow):
    preview_preprocessor: ProcessorBase = None

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
        action_use_initial_image = actions.use_initial_image.create(self)
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
        image_menu.addAction(action_use_initial_image)
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
        action_use_initial_image.triggered.connect(lambda: self.on_use_initial_image(self.image_viewer.metadata))
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

        txt2img_button = actions.txt2img.tool_button()
        img2img_button = actions.img2img.tool_button()

        mode_toolbar.addWidget(txt2img_button)
        mode_toolbar.addWidget(img2img_button)

        self.button_group = QButtonGroup()
        self.button_group.addButton(txt2img_button, 0)
        self.button_group.addButton(img2img_button, 1)
        self.button_group.idToggled.connect(self.on_mode_changed)

        # Configuration controls
        self.config_frame = QFrame()
        self.config_frame.setContentsMargins(0, 0, 0, 0)
        
        config_scroll_area = QScrollArea()
        config_scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        config_scroll_area.setWidgetResizable(True)
        config_scroll_area.setWidget(self.config_frame)
        config_scroll_area.setFocusPolicy(Qt.NoFocus)

        self.model_combo_box = QComboBox()
        self.settings.beginGroup('Models')
        for key in self.settings.childKeys():
            value = self.settings.value(key)
            index = self.model_combo_box.addItem(key, value)
        self.settings.endGroup()
        index = self.model_combo_box.findData(self.settings.value('model'))
        if index != -1:
            self.model_combo_box.setCurrentIndex(index)

        self.prompt_edit = PromptTextEdit(8, 'Prompt')
        self.prompt_edit.setPlainText(self.settings.value('prompt'))
        self.prompt_edit.return_pressed.connect(self.on_generate_image)
        self.negative_prompt_edit = PromptTextEdit(5, 'Negative Prompt')
        self.negative_prompt_edit.setPlainText(self.settings.value('negative_prompt'))
        self.negative_prompt_edit.return_pressed.connect(self.on_generate_image)

        self.generate_button = QPushButton('Generate')
        self.generate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.generate_button.clicked.connect(self.on_generate_image)

        cancel_button = QPushButton()
        cancel_button.setIcon(QIcon(utils.resource_path('cancel_icon.png')))
        cancel_button.setToolTip('Cancel')
        cancel_button.clicked.connect(self.on_cancel_generation)

        controls_frame = QFrame()
        controls_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        num_images_label = QLabel('Images')
        num_images_label.setAlignment(Qt.AlignCenter)
        self.num_images_spin_box = QSpinBox()
        self.num_images_spin_box.setAlignment(Qt.AlignCenter)
        self.num_images_spin_box.setFixedWidth(80)
        self.num_images_spin_box.setMinimum(1)
        self.num_images_spin_box.setValue(int(self.settings.value('num_images_per_prompt')))
        num_steps_label = QLabel('Steps')
        num_steps_label.setAlignment(Qt.AlignCenter)
        self.num_steps_spin_box = QSpinBox()
        self.num_steps_spin_box.setAlignment(Qt.AlignCenter)
        self.num_steps_spin_box.setFixedWidth(80)
        self.num_steps_spin_box.setMinimum(1)
        self.num_steps_spin_box.setValue(int(self.settings.value('num_inference_steps')))
        guidance_scale_label = QLabel('CFG Scale')
        guidance_scale_label.setAlignment(Qt.AlignCenter)
        self.guidance_scale_spin_box = QDoubleSpinBox()
        self.guidance_scale_spin_box.setAlignment(Qt.AlignCenter)
        self.guidance_scale_spin_box.setFixedWidth(80)
        self.guidance_scale_spin_box.setSingleStep(0.5)
        self.guidance_scale_spin_box.setMinimum(1.0)
        self.guidance_scale_spin_box.setValue(float(self.settings.value('guidance_scale')))
        width_label = QLabel('Width')
        width_label.setAlignment(Qt.AlignCenter)
        self.width_spin_box = QSpinBox()
        self.width_spin_box.setAlignment(Qt.AlignCenter)
        self.width_spin_box.setFixedWidth(80)
        self.width_spin_box.setSingleStep(64)
        self.width_spin_box.setMinimum(64)
        self.width_spin_box.setMaximum(1024)
        self.width_spin_box.setValue(int(self.settings.value('width')))
        height_label = QLabel('Height')
        height_label.setAlignment(Qt.AlignCenter)
        self.height_spin_box = QSpinBox()
        self.height_spin_box.setAlignment(Qt.AlignCenter)
        self.height_spin_box.setFixedWidth(80)
        self.height_spin_box.setSingleStep(64)
        self.height_spin_box.setMinimum(64)
        self.height_spin_box.setMaximum(1024)
        self.height_spin_box.setValue(int(self.settings.value('height')))
        scheduler_label = QLabel('Scheduler')
        scheduler_label.setAlignment(Qt.AlignCenter)
        self.scheduler_combo_box = QComboBox()
        self.scheduler_combo_box.addItems(configuration.schedulers.keys())
        self.scheduler_combo_box.setFixedWidth(120)
        self.scheduler_combo_box.setCurrentText(self.settings.value('scheduler'))

        self.manual_seed_check_box = QCheckBox('Manual Seed')

        self.seed_frame = QFrame()
        self.seed_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.seed_lineedit = QLineEdit()
        self.seed_lineedit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.seed_lineedit.setText(str(self.settings.value('seed')))
        seed_random_button = QPushButton('New')
        seed_random_button.clicked.connect(self.on_seed_random_clicked)

        manual_seed = self.settings.value('manual_seed', type=bool)
        self.seed_frame.setEnabled(manual_seed)
        self.manual_seed_check_box.setChecked(manual_seed)
        self.manual_seed_check_box.stateChanged.connect(self.on_manual_seed_check_box_changed)

        self.condition_frame = QFrame()
        conditions_label = QLabel('Condition: ')
        self.condition_combo_box = QComboBox()
        self.condition_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.condition_combo_box.addItems(configuration.conditions.keys())
        self.condition_combo_box.setCurrentText(self.settings.value('condition'))
        self.condition_combo_box.currentIndexChanged.connect(self.on_condition_combobox_value_changed)

        self.control_net_frame = QFrame()
        self.control_net_preprocess_check_box = QCheckBox('Preprocess')
        self.control_net_preprocess_check_box.setChecked(self.settings.value('control_net_preprocess', type=bool))
        self.control_net_preview_preprocessor_button = QPushButton('Preview')
        self.control_net_preview_preprocessor_button.clicked.connect(self.on_control_net_preview_preprocessor_button_clicked)

        control_net_model_label = QLabel('Model')
        control_net_model_label.setAlignment(Qt.AlignCenter)
        self.control_net_model_combo_box = QComboBox()

        self.control_net_scale = FloatSliderSpinBox('ControlNet Scale', float(self.settings.value('control_net_scale')))

        control_net_grid = QGridLayout()
        control_net_grid.setContentsMargins(0, 0, 0, 0)
        control_net_grid.setVerticalSpacing(2)
        control_net_grid.addWidget(self.control_net_preprocess_check_box, 0, 0)
        control_net_grid.setAlignment(self.control_net_preprocess_check_box, Qt.AlignCenter)
        control_net_grid.addWidget(self.control_net_preview_preprocessor_button, 1, 0)
        control_net_grid.addWidget(control_net_model_label, 0, 1)
        control_net_grid.addWidget(self.control_net_model_combo_box, 1, 1)

        self.img_strength = FloatSliderSpinBox('Image Strength', float(self.settings.value('img_strength')))

        upscale_enabled = self.settings.value('upscale_enabled', type=bool)
        self.upscale_enabled_check_box = QCheckBox('Upscaling')
        self.upscale_enabled_check_box.setChecked(upscale_enabled)
        self.upscale_enabled_check_box.stateChanged.connect(self.on_upscale_enabled_check_box_changed)
        self.upscale_frame = QFrame()
        self.upscale_frame.setEnabled(upscale_enabled)
        upscale_factor_label = QLabel('Factor: ')
        self.upscale_factor_combo_box = QComboBox()
        self.upscale_factor_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.upscale_factor_combo_box.addItem('2x', 2)
        self.upscale_factor_combo_box.addItem('4x', 4)
        index = self.upscale_factor_combo_box.findData(self.settings.value('upscale_factor', type=int))
        if index != -1:
            self.upscale_factor_combo_box.setCurrentIndex(index)        
        self.upscale_denoising_strength = FloatSliderSpinBox('Denoising Strength', float(self.settings.value('upscale_denoising_strength')))
        self.upscale_blend_strength = FloatSliderSpinBox('Upscale Strength', float(self.settings.value('upscale_denoising_strength')))
        self.face_strength = FloatSliderSpinBox('Face Restoration', float(self.settings.value('face_blend_strength')), checkable=True)
        self.face_strength.check_box.setChecked(self.settings.value('face_enabled', type=bool))

        generate_hlayout = QHBoxLayout()
        generate_hlayout.setContentsMargins(0, 0, 0, 0)
        generate_hlayout.addWidget(self.generate_button)
        generate_hlayout.addWidget(cancel_button)

        controls_grid = QGridLayout(controls_frame)
        controls_grid.setContentsMargins(0, 0, 0, 0)
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

        seed_hlayout = QHBoxLayout(self.seed_frame)
        seed_hlayout.setContentsMargins(0, 0, 0, 0)
        seed_hlayout.addWidget(self.seed_lineedit)
        seed_hlayout.addWidget(seed_random_button)

        seed_vlayout = QVBoxLayout()
        seed_vlayout.setContentsMargins(0, 0, 0, 0) 
        seed_vlayout.setSpacing(0)
        seed_check_box_layout = QHBoxLayout()
        seed_check_box_layout.setAlignment(Qt.AlignCenter)
        seed_check_box_layout.addWidget(self.manual_seed_check_box)
        seed_vlayout.addLayout(seed_check_box_layout)
        seed_vlayout.addWidget(self.seed_frame)

        condition_frame_layout = QHBoxLayout(self.condition_frame)
        condition_frame_layout.setContentsMargins(0, 0, 0, 0)
        condition_frame_layout.addWidget(conditions_label)
        condition_frame_layout.addWidget(self.condition_combo_box)

        control_net_layout = QVBoxLayout(self.control_net_frame)
        control_net_layout.setContentsMargins(0, 0, 0, 0)
        control_net_layout.setSpacing(0)
        control_net_layout.addLayout(control_net_grid)
        control_net_layout.addWidget(self.control_net_scale)
        control_net_layout.addWidget(utils.horizontal_separator())

        condition_layout = QVBoxLayout()
        condition_layout.setContentsMargins(0, 0, 0, 0) 
        condition_layout.setSpacing(0)
        condition_layout.addWidget(self.condition_frame)
        condition_layout.addWidget(self.control_net_frame)
        condition_layout.addWidget(self.img_strength)

        upscale_factor_layout = QHBoxLayout()
        upscale_factor_layout.setContentsMargins(0, 0, 0, 0) 
        upscale_factor_layout.setSpacing(0)
        upscale_factor_layout.addWidget(upscale_factor_label)
        upscale_factor_layout.addWidget(self.upscale_factor_combo_box)

        upscale_frame_layout = QVBoxLayout(self.upscale_frame)
        upscale_frame_layout.setContentsMargins(0, 0, 0, 0)
        upscale_frame_layout.setSpacing(0)
        upscale_frame_layout.addLayout(upscale_factor_layout)
        upscale_frame_layout.addWidget(self.upscale_denoising_strength)
        upscale_frame_layout.addWidget(self.upscale_blend_strength)

        upscale_enabled_check_box_layout = QVBoxLayout()
        upscale_enabled_check_box_layout.setContentsMargins(0, 0, 0, 0) 
        upscale_enabled_check_box_layout.setSpacing(0)
        upscale_enabled_check_box_layout.addWidget(self.upscale_enabled_check_box)
        upscale_enabled_check_box_layout.setAlignment(self.upscale_enabled_check_box, Qt.AlignCenter)

        upscale_layout = QVBoxLayout()
        upscale_layout.setContentsMargins(0, 0, 0, 0) 
        upscale_layout.setSpacing(0)
        upscale_layout.addLayout(upscale_enabled_check_box_layout)
        upscale_layout.addWidget(self.upscale_frame)

        config_layout = QVBoxLayout(self.config_frame)
        config_layout.setContentsMargins(0, 0, 0, 0) 
        config_layout.addWidget(self.model_combo_box)
        config_layout.addWidget(self.prompt_edit)
        config_layout.addWidget(self.negative_prompt_edit)
        config_layout.addLayout(generate_hlayout)
        config_layout.addWidget(utils.horizontal_separator())
        config_layout.addWidget(controls_frame)
        config_layout.addWidget(utils.horizontal_separator())
        config_layout.addLayout(seed_vlayout)
        config_layout.addWidget(utils.horizontal_separator())
        config_layout.addLayout(condition_layout)
        config_layout.addLayout(upscale_layout)
        config_layout.addWidget(utils.horizontal_separator())
        config_layout.addWidget(self.face_strength)
        config_layout.addStretch()

        # Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.locate_source_button.pressed.connect(lambda: self.thumbnail_viewer.select_image(self.image_viewer.left_image_path()))
        self.image_viewer.send_to_img2img_button.pressed.connect(lambda: self.on_send_to_img2img(self.image_viewer.metadata))
        self.image_viewer.use_prompt_button.pressed.connect(lambda: self.on_use_prompt(self.image_viewer.metadata))
        self.image_viewer.use_seed_button.pressed.connect(lambda: self.on_use_seed(self.image_viewer.metadata))
        self.image_viewer.use_initial_image_button.pressed.connect(lambda: self.on_use_initial_image(self.image_viewer.metadata))
        self.image_viewer.use_all_button.pressed.connect(lambda: self.on_use_all(self.image_viewer.metadata))
        self.image_viewer.delete_button.pressed.connect(lambda: self.on_delete(self.image_viewer.metadata))

        # Thumbnail viewer
        self.thumbnail_viewer = ThumbnailViewer(self.settings, self.collections)
        self.thumbnail_viewer.file_dropped.connect(self.on_thumbnail_file_dropped)
        self.thumbnail_viewer.list_widget.itemSelectionChanged.connect(self.on_thumbnail_selection_change)
        self.thumbnail_viewer.action_send_to_img2img.triggered.connect(lambda: self.on_send_to_img2img(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_use_prompt.triggered.connect(lambda: self.on_use_prompt(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_use_seed.triggered.connect(lambda: self.on_use_seed(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_use_initial_image.triggered.connect(lambda: self.on_use_initial_image(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_use_all.triggered.connect(lambda: self.on_use_all(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_delete.triggered.connect(lambda: self.on_delete(self.thumbnail_viewer.get_current_metadata()))
        self.thumbnail_viewer.action_reveal_in_finder.triggered.connect(lambda: self.on_reveal_in_finder(self.thumbnail_viewer.get_current_metadata()))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.image_viewer)
        splitter.addWidget(self.thumbnail_viewer)
        splitter.setStretchFactor(0, 1)  # left widget
        splitter.setStretchFactor(1, 0)  # right widget

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setMinimum(0)

        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(8, 2, 8, 8)
        hlayout.setSpacing(8)
        hlayout.addWidget(config_scroll_area)
        hlayout.addWidget(splitter)

        vlayout = QVBoxLayout(central_widget)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        vlayout.addWidget(self.progress_bar)
        vlayout.addLayout(hlayout)

        self.setWindowTitle(configuration.APP_NAME)
        self.setGeometry(100, 100, 1200, 600)

        # Apply settings that impact other controls
        if self.settings.value('source_path') != '':
            self.image_viewer.set_left_image(self.settings.value('source_path'))
        self.set_type(self.settings.value('type'))
        self.on_thumbnail_selection_change()
        self.on_condition_combobox_value_changed(self.condition_combo_box.currentIndex())

    def show_about_dialog(self):
        about_dialog = AboutDialog()
        about_dialog.exec()

    def show_preferences_dialog(self):
        dialog = PreferencesDialog(self)
        dialog.exec()
        
    def set_type(self, type):
        self.type = type
        if self.type == 'txt2img':
            self.button_group.button(0).setChecked(True)
        elif self.type == 'img2img':
            self.button_group.button(1).setChecked(True)

    def on_mode_changed(self, button_id, checked):
        if not checked:
            return
        if button_id == 0:
            self.type = 'txt2img'
        elif button_id == 1:
            self.type = 'img2img'
        
        self.update_control_visibility()

    def on_cancel_generation(self):
        if self.generate_thread:
            self.generate_thread.cancel = True

    def on_condition_combobox_value_changed(self, index):
        condition_name = self.condition_combo_box.itemText(index)
        condition = configuration.conditions.get(condition_name, None)
        if isinstance(condition, Img2ImgCondition):
            pass
        elif isinstance(condition, ControlNetCondition):
            self.control_net_model_combo_box.clear()
            for key, value in condition.models.items():
                self.control_net_model_combo_box.addItem(key, value)
            self.control_net_model_combo_box.setCurrentText(self.settings.value('control_net_model'))

        self.update_control_visibility()

    def on_control_net_preview_preprocessor_button_clicked(self):
        source_path = self.image_viewer.left_image_path()
        width = self.width_spin_box.value()
        height = self.height_spin_box.value()

        full_path = os.path.join(configuration.IMAGES_PATH, source_path)
        with Image.open(full_path) as image:
            image = image.convert('RGB')
            image = image.resize((width, height))
            source_image = image.copy()

        condition_name = self.condition_combo_box.currentText()
        condition = configuration.conditions.get(condition_name, None)
        if isinstance(condition, ControlNetCondition):
            if not isinstance(self.preview_preprocessor, condition.preprocessor):
                self.preview_preprocessor = condition.preprocessor()
            source_image = self.preview_preprocessor(source_image)
            if self.settings.value('reduce_memory', type=bool):
                self.preview_preprocessor = None
            output_path = 'preprocessed.png'
            full_path = os.path.join(configuration.IMAGES_PATH, output_path)
            source_image.save(full_path)
            self.image_viewer.set_right_image(output_path)

    def update_control_visibility(self):
        if self.type == 'txt2img':
            self.condition_frame.setVisible(False)
            self.img_strength.setVisible(False)
            self.control_net_frame.setVisible(False)
            self.image_viewer.set_both_images_visible(False)
        elif self.type == 'img2img':
            condition_name = self.condition_combo_box.currentText()
            condition = configuration.conditions.get(condition_name, None)
            self.condition_frame.setVisible(True)
            if isinstance(condition, Img2ImgCondition):
                self.img_strength.setVisible(True)
                self.control_net_frame.setVisible(False)
            elif isinstance(condition, ControlNetCondition):
                self.img_strength.setVisible(False)
                self.control_net_frame.setVisible(True)
            self.image_viewer.set_both_images_visible(True)

        self.config_frame.adjustSize()

    def on_generate_image(self):
        if not self.manual_seed_check_box.isChecked():
            self.randomize_seed()

        condition_name = self.condition_combo_box.currentText()
        condition = configuration.conditions.get(condition_name, None)

        self.settings.setValue('collection', self.thumbnail_viewer.collection())
        self.settings.setValue('type', self.type)
        self.settings.setValue('model', self.model_combo_box.currentData())
        self.settings.setValue('scheduler', self.scheduler_combo_box.currentText())
        self.settings.setValue('prompt', self.prompt_edit.toPlainText())
        self.settings.setValue('negative_prompt', self.negative_prompt_edit.toPlainText())
        self.settings.setValue('manual_seed', self.manual_seed_check_box.isChecked())
        self.settings.setValue('seed', self.seed_lineedit.text())
        self.settings.setValue('num_images_per_prompt', self.num_images_spin_box.value())
        self.settings.setValue('num_inference_steps', self.num_steps_spin_box.value())
        self.settings.setValue('guidance_scale', self.guidance_scale_spin_box.value())
        self.settings.setValue('width', self.width_spin_box.value())
        self.settings.setValue('height', self.height_spin_box.value())
        self.settings.setValue('condition', condition_name)
        if isinstance(condition, ControlNetCondition):
            self.settings.setValue('control_net_preprocess', self.control_net_preprocess_check_box.isChecked())
            self.settings.setValue('control_net_model', self.control_net_model_combo_box.currentData())
            self.settings.setValue('control_net_scale', self.control_net_scale.spin_box.value())
        self.settings.setValue('source_path', self.image_viewer.left_image_path())
        self.settings.setValue('img_strength', self.img_strength.spin_box.value())
        self.settings.setValue('upscale_enabled', self.upscale_enabled_check_box.isChecked())
        self.settings.setValue('upscale_factor', self.upscale_factor_combo_box.currentData())
        self.settings.setValue('upscale_denoising_strength', self.upscale_denoising_strength.spin_box.value())
        self.settings.setValue('upscale_blend_strength', self.upscale_blend_strength.spin_box.value())
        self.settings.setValue('face_enabled', self.face_strength.check_box.isChecked())
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
            self.progress_bar.setStyleSheet('QProgressBar:chunk { background-color: grey; }')
        else:
            self.progress_bar.setStyleSheet('QProgressBar:chunk { background-color: blue; }')
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

    def on_manual_seed_check_box_changed(self, state):
        self.seed_frame.setEnabled(state)

    def on_seed_random_clicked(self):
        self.randomize_seed()

    def on_upscale_enabled_check_box_changed(self, state):
        self.upscale_frame.setEnabled(state)

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
            self.image_viewer.set_left_image(image_metadata.path)
            self.set_type('img2img')
    
    def on_use_prompt(self, image_metadata):
        if image_metadata is not None:
            self.prompt_edit.setPlainText(image_metadata.prompt)
            self.negative_prompt_edit.setPlainText(image_metadata.negative_prompt)

    def on_use_seed(self, image_metadata):
        if image_metadata is not None:
            self.manual_seed_check_box.setChecked(True)
            self.seed_lineedit.setText(str(image_metadata.seed))

    def on_use_initial_image(self, image_metadata):
        if image_metadata is not None:
            self.image_viewer.set_left_image(image_metadata.source_path)
            self.img_strength.spin_box.setValue(image_metadata.img_strength)
            self.set_type('img2img')
 
    def on_use_all(self, image_metadata):
        if image_metadata is not None:
            self.prompt_edit.setPlainText(image_metadata.prompt)
            self.negative_prompt_edit.setPlainText(image_metadata.negative_prompt)
            self.manual_seed_check_box.setChecked(True)
            self.seed_lineedit.setText(str(image_metadata.seed))
            self.num_steps_spin_box.setValue(image_metadata.num_inference_steps)
            self.guidance_scale_spin_box.setValue(image_metadata.guidance_scale)
            self.width_spin_box.setValue(image_metadata.width)
            self.height_spin_box.setValue(image_metadata.height)
            self.scheduler_combo_box.setCurrentText(image_metadata.scheduler)
            if image_metadata.type == 'img2img':
                self.image_viewer.set_left_image(image_metadata.source_path)
                self.condition_combo_box.setCurrentText(image_metadata.condition)
                condition = configuration.conditions.get(image_metadata.condition, None)
                if isinstance(condition, Img2ImgCondition):
                    self.img_strength.spin_box.setValue(image_metadata.img_strength)
                if isinstance(condition, ControlNetCondition):
                    self.control_net_preprocess_check_box.setChecked(image_metadata.control_net_preprocess)
                    self.control_net_model_combo_box.setCurrentText(image_metadata.control_net_model)

            if image_metadata.upscale_enabled:
                self.upscale_enabled_check_box.setChecked(True)
                index = self.upscale_factor_combo_box.findData(image_metadata.upscale_factor)
                if index != -1:
                    self.upscale_factor_combo_box.setCurrentIndex(index)
                self.upscale_denoising_strength.spin_box.setValue(image_metadata.upscale_denoising_strength)
                self.upscale_blend_strength.spin_box.setValue(image_metadata.upscale_blend_strength)
            else:
                self.upscale_enabled_check_box.setChecked(False)

            if image_metadata.face_enabled:
                self.face_strength.check_box.setChecked(True)
                self.face_strength.spin_box.setValue(image_metadata.face_blend_strength)
            else:
                self.face_strength.check_box.setChecked(False)

            self.set_type(image_metadata.type)

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
        self.thumbnail_viewer.list_widget.setCurrentRow(0)
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
