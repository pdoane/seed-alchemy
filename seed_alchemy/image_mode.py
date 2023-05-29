import json
import os
import random
import shutil
import sys

import torch
from clip_interrogator import Config, Interrogator
from PIL import Image, PngImagePlugin
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import actions, configuration, control_net_config
from . import font_awesome as fa
from . import utils
from .delete_image_dialog import DeleteImageDialog
from .generate_thread import GenerateThread
from .image_history import ImageHistory
from .image_metadata import ControlNetConditionMetadata, ImageMetadata
from .image_viewer import ImageViewer
from .processors import ProcessorBase
from .prompt_text_edit import PromptTextEdit
from .thumbnail_loader import ThumbnailLoader
from .thumbnail_model import ThumbnailModel
from .thumbnail_viewer import ThumbnailViewer
from .widgets import (
    ComboBox,
    DoubleSpinBox,
    FloatSliderSpinBox,
    FrameWithCloseButton,
    IntSliderSpinBox,
    ScrollArea,
    SpinBox,
)
from .icon_engine import FontAwesomeIconEngine

if sys.platform == "darwin":
    from AppKit import NSApplication


class SourceImageUI:
    def __init__(self):
        self.frame: QFrame = None
        self.label: QLabel = None
        self.line_edit: QLineEdit = None
        self.context_menu: QMenu = None
        self.previous_image_path: str = None


class ControlNetConditionUI:
    frame: FrameWithCloseButton
    model_combo_box: ComboBox
    preprocessor_combo_box: ComboBox
    source_image_ui: SourceImageUI
    params_layout_index: int
    params: list[QWidget]
    scale: FloatSliderSpinBox


class ImageModeWidget(QWidget):
    preview_preprocessor: ProcessorBase = None
    img2img_source_ui: SourceImageUI = None
    condition_uis: list[ControlNetConditionUI] = []
    override_metadata = None

    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.progress_bar = main_window.progress_bar
        self.settings: QSettings = main_window.settings
        self.collections = main_window.collections
        self.image_history = ImageHistory()
        self.image_history.current_image_changed.connect(self.on_current_image_changed)
        self.thumbnail_loader = ThumbnailLoader()
        QApplication.instance().aboutToQuit.connect(self.thumbnail_loader.shutdown)
        self.thumbnail_model = ThumbnailModel(self.thumbnail_loader, 100)

        self.generate_thread = None

        # Set as Source Menu
        self.set_as_source_menu = QMenu("Set as Source", self)
        self.set_as_source_menu.setIcon(QIcon(FontAwesomeIconEngine(fa.icon_share)))
        self.set_as_source_menu.addAction(QAction("Dummy...", self))
        self.set_as_source_menu.aboutToShow.connect(self.populate_set_as_source_menu)

        # Move To Menu
        self.move_image_menu = QMenu("Move To", self)
        self.move_image_menu.setIcon(utils.empty_qicon())
        for collection in self.collections:
            item_action = QAction(collection, self)
            item_action.triggered.connect(lambda checked=False, collection=collection: self.on_move_image(collection))
            self.move_image_menu.addAction(item_action)

        # Generate
        self.generate_button = QPushButton("Generate")
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

        self.config_scroll_area = ScrollArea()
        self.config_scroll_area.setFrameStyle(QFrame.NoFrame)
        self.config_scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.config_scroll_area.setWidgetResizable(True)
        self.config_scroll_area.setWidget(self.config_frame)
        self.config_scroll_area.setFocusPolicy(Qt.NoFocus)

        # Model
        model_label = QLabel("Stable Diffusion Model")
        model_label.setAlignment(Qt.AlignCenter)
        self.model_combo_box = ComboBox()
        self.model_combo_box.addItems(configuration.stable_diffusion_models.keys())

        # Prompts
        self.prompt_edit = PromptTextEdit(8, "Prompt")
        self.prompt_edit.return_pressed.connect(self.on_generate_image)
        self.negative_prompt_edit = PromptTextEdit(5, "Negative Prompt")
        self.negative_prompt_edit.return_pressed.connect(self.on_generate_image)

        self.prompt_group_box = QGroupBox("Prompts")
        prompt_group_box_layout = QVBoxLayout(self.prompt_group_box)
        prompt_group_box_layout.addWidget(self.prompt_edit)
        prompt_group_box_layout.addWidget(self.negative_prompt_edit)

        # General
        num_images_label = QLabel("Images")
        num_images_label.setAlignment(Qt.AlignCenter)
        self.num_images_spin_box = SpinBox()
        self.num_images_spin_box.setAlignment(Qt.AlignCenter)
        self.num_images_spin_box.setFixedWidth(80)
        self.num_images_spin_box.setMinimum(1)
        num_steps_label = QLabel("Steps")
        num_steps_label.setAlignment(Qt.AlignCenter)
        self.num_steps_spin_box = SpinBox()
        self.num_steps_spin_box.setAlignment(Qt.AlignCenter)
        self.num_steps_spin_box.setFixedWidth(80)
        self.num_steps_spin_box.setMinimum(1)
        guidance_scale_label = QLabel("CFG Scale")
        guidance_scale_label.setAlignment(Qt.AlignCenter)
        self.guidance_scale_spin_box = DoubleSpinBox()
        self.guidance_scale_spin_box.setFocusPolicy(Qt.StrongFocus)
        self.guidance_scale_spin_box.setAlignment(Qt.AlignCenter)
        self.guidance_scale_spin_box.setFixedWidth(80)
        self.guidance_scale_spin_box.setSingleStep(0.5)
        self.guidance_scale_spin_box.setMinimum(1.0)
        width_label = QLabel("Width")
        width_label.setAlignment(Qt.AlignCenter)
        self.width_spin_box = SpinBox()
        self.width_spin_box.setAlignment(Qt.AlignCenter)
        self.width_spin_box.setFixedWidth(80)
        self.width_spin_box.setSingleStep(64)
        self.width_spin_box.setMinimum(64)
        self.width_spin_box.setMaximum(2048)
        height_label = QLabel("Height")
        height_label.setAlignment(Qt.AlignCenter)
        self.height_spin_box = SpinBox()
        self.height_spin_box.setAlignment(Qt.AlignCenter)
        self.height_spin_box.setFixedWidth(80)
        self.height_spin_box.setSingleStep(64)
        self.height_spin_box.setMinimum(64)
        self.height_spin_box.setMaximum(2048)
        scheduler_label = QLabel("Scheduler")
        scheduler_label.setAlignment(Qt.AlignCenter)
        self.scheduler_combo_box = ComboBox()
        self.scheduler_combo_box.addItems(configuration.schedulers.keys())
        self.scheduler_combo_box.setFixedWidth(120)

        self.general_group_box = QGroupBox("General")

        model_vlayout = QVBoxLayout()
        model_vlayout.setContentsMargins(0, 0, 0, 0)
        model_vlayout.setSpacing(2)
        model_vlayout.addWidget(model_label)
        model_vlayout.addWidget(self.model_combo_box)

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
        controls_vlayout.addLayout(model_vlayout)
        controls_vlayout.addLayout(controls_grid)

        # Seed
        self.seed_line_edit = QLineEdit()
        self.seed_line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        seed_random_button = QPushButton("New")
        seed_random_button.clicked.connect(self.on_seed_random_clicked)

        self.seed_frame = QFrame()
        self.seed_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        seed_hlayout = QHBoxLayout(self.seed_frame)
        seed_hlayout.setContentsMargins(0, 0, 0, 0)
        seed_hlayout.addWidget(self.seed_line_edit)
        seed_hlayout.addWidget(seed_random_button)

        self.manual_seed_group_box = QGroupBox("Manual Seed")
        self.manual_seed_group_box.setCheckable(True)
        manual_seed_group_box_layout = QVBoxLayout(self.manual_seed_group_box)
        manual_seed_group_box_layout.addWidget(self.seed_frame)

        # Image to Image
        self.img2img_source_ui = self.create_source_image_ui()

        self.img2img_noise = FloatSliderSpinBox("Noise")

        self.img2img_group_box = QGroupBox("Image to Image")
        self.img2img_group_box.setCheckable(True)

        img2img_group_box_layout = QVBoxLayout(self.img2img_group_box)
        img2img_group_box_layout.addWidget(self.img2img_source_ui.frame)
        img2img_group_box_layout.addWidget(self.img2img_noise)

        # ControlNet
        self.control_net_guidance_start = FloatSliderSpinBox("Guidance Start")
        self.control_net_guidance_end = FloatSliderSpinBox("Guidance End")

        self.control_net_add_button = QPushButton("Add Condition")
        self.control_net_add_button.clicked.connect(self.on_add_control_net)

        self.control_net_group_box = QGroupBox("Control Net")
        self.control_net_group_box.setCheckable(True)
        self.control_net_group_box_layout = QVBoxLayout(self.control_net_group_box)
        self.control_net_group_box_layout.addWidget(self.control_net_guidance_start)
        self.control_net_group_box_layout.addWidget(self.control_net_guidance_end)
        self.control_net_dynamic_index = self.control_net_group_box_layout.count()
        self.control_net_group_box_layout.addWidget(self.control_net_add_button)

        # Upscale
        upscale_factor_label = QLabel("Factor: ")
        self.upscale_factor_combo_box = ComboBox()
        self.upscale_factor_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.upscale_factor_combo_box.addItem("2x", 2)
        self.upscale_factor_combo_box.addItem("4x", 4)
        self.upscale_denoising = FloatSliderSpinBox("Denoising")
        self.upscale_blend = FloatSliderSpinBox("Strength")

        self.upscale_group_box = QGroupBox("Upscaling")
        self.upscale_group_box.setCheckable(True)
        upscale_factor_layout = QHBoxLayout()
        upscale_factor_layout.setContentsMargins(0, 0, 0, 0)
        upscale_factor_layout.setSpacing(0)
        upscale_factor_layout.addWidget(upscale_factor_label)
        upscale_factor_layout.addWidget(self.upscale_factor_combo_box)
        upscale_group_box_layout = QVBoxLayout(self.upscale_group_box)
        upscale_group_box_layout.addLayout(upscale_factor_layout)
        upscale_group_box_layout.addWidget(self.upscale_denoising)
        upscale_group_box_layout.addWidget(self.upscale_blend)

        # Face Restoration
        self.face_blend = FloatSliderSpinBox("Strength")

        self.face_restoration_group_box = QGroupBox("Face Restoration")
        self.face_restoration_group_box.setCheckable(True)

        face_restoration_group_box_layout = QVBoxLayout(self.face_restoration_group_box)
        face_restoration_group_box_layout.addWidget(self.face_blend)

        # High Resolution
        self.high_res_factor = FloatSliderSpinBox("Factor", minimum=1, maximum=2)
        self.high_res_steps = IntSliderSpinBox("Steps", minimum=1, maximum=200)
        self.high_res_guidance_scale = FloatSliderSpinBox("Guidance", minimum=1, maximum=50, step=0.5)
        self.high_res_noise = FloatSliderSpinBox("Noise")

        self.high_res_group_box = QGroupBox("High Resolution")
        self.high_res_group_box.setCheckable(True)

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
        config_layout.addWidget(self.face_restoration_group_box)
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

        image_mode_config_vlayout = QVBoxLayout()
        image_mode_config_vlayout.addLayout(generate_hlayout)
        image_mode_config_vlayout.addWidget(self.config_scroll_area)

        image_mode_layout = QHBoxLayout(self)
        image_mode_layout.setContentsMargins(8, 2, 8, 8)
        image_mode_layout.setSpacing(8)
        image_mode_layout.addLayout(image_mode_config_vlayout)
        image_mode_layout.addWidget(image_mode_splitter)

        # Set current values
        self.model_combo_box.setCurrentText(self.settings.value("model"))
        self.prompt_edit.setPlainText(self.settings.value("prompt"))
        self.negative_prompt_edit.setPlainText(self.settings.value("negative_prompt"))
        self.num_images_spin_box.setValue(int(self.settings.value("num_images_per_prompt")))
        self.num_steps_spin_box.setValue(int(self.settings.value("num_inference_steps")))
        self.guidance_scale_spin_box.setValue(float(self.settings.value("guidance_scale")))
        self.width_spin_box.setValue(int(self.settings.value("width")))
        self.height_spin_box.setValue(int(self.settings.value("height")))
        self.scheduler_combo_box.setCurrentText(self.settings.value("scheduler"))
        self.seed_line_edit.setText(str(self.settings.value("seed")))
        self.manual_seed_group_box.setChecked(self.settings.value("manual_seed", type=bool))
        self.img2img_source_ui.line_edit.setText(self.settings.value("img2img_source", type=str))
        self.img2img_noise.setValue(float(self.settings.value("img2img_noise")))
        self.img2img_group_box.setChecked(self.settings.value("img2img_enabled", type=bool))
        self.control_net_guidance_start.setValue(self.settings.value("control_net_guidance_start", type=float))
        self.control_net_guidance_end.setValue(self.settings.value("control_net_guidance_end", type=float))
        self.control_net_group_box.setChecked(self.settings.value("control_net_enabled", type=bool))
        condition_metas = [
            ControlNetConditionMetadata.from_dict(item)
            for item in json.loads(self.settings.value("control_net_conditions"))
        ]
        for i, condition_meta in enumerate(condition_metas):
            condition_ui = self.create_control_net_ui(condition_meta)
            self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, condition_ui.frame)
            self.condition_uis.append(condition_ui)
        utils.set_current_data(self.upscale_factor_combo_box, self.settings.value("upscale_factor", type=int))
        self.upscale_denoising.setValue(float(self.settings.value("upscale_denoising")))
        self.upscale_blend.setValue(float(self.settings.value("upscale_blend")))
        self.upscale_group_box.setChecked(self.settings.value("upscale_enabled", type=bool))
        self.face_blend.setValue(float(self.settings.value("face_blend")))
        self.face_restoration_group_box.setChecked(self.settings.value("face_enabled", type=bool))
        self.high_res_factor.setValue(self.settings.value("high_res_factor", type=float))
        self.high_res_steps.setValue(self.settings.value("high_res_steps", type=int))
        self.high_res_guidance_scale.setValue(self.settings.value("high_res_guidance_scale", type=float))
        self.high_res_noise.setValue(self.settings.value("high_res_noise", type=float))
        self.high_res_group_box.setChecked(self.settings.value("high_res_enabled", type=bool))

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
        action_interrogate = actions.interrogate.create(self)
        action_interrogate.triggered.connect(self.on_interrogate)
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
        image_menu.addAction(action_interrogate)
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

    def update_config_frame_size(self):
        self.config_frame.adjustSize()

        # Workaround QT issue where the size of QScrollArea.widget() is cached and not updated
        scroll_pos = self.config_scroll_area.verticalScrollBar().value()
        self.config_scroll_area.takeWidget()
        self.config_scroll_area.setWidget(self.config_frame)
        self.config_scroll_area.updateGeometry()
        self.config_scroll_area.verticalScrollBar().setValue(scroll_pos)

    def get_current_metadata(self):
        if self.override_metadata is not None:
            return self.override_metadata
        else:
            return self.image_viewer.metadata

    def set_override_metadata(self, metadata):
        self.override_metadata = metadata

    def create_source_image_ui(self) -> SourceImageUI:
        source_image_ui = SourceImageUI()

        source_image_ui.frame = QFrame()
        source_image_ui.frame.setContentsMargins(0, 0, 0, 0)

        source_label = QLabel("Source")
        source_label.setAlignment(Qt.AlignCenter)

        source_image_ui.label = QLabel()
        source_image_ui.label.setFrameStyle(QFrame.Box)
        source_image_ui.label.setContextMenuPolicy(Qt.CustomContextMenu)
        source_image_ui.label.setFixedSize(96, 96)
        source_image_ui.label.customContextMenuRequested.connect(
            lambda point: self.show_source_image_context_menu(source_image_ui, point)
        )

        source_image_ui.line_edit = QLineEdit()
        source_image_ui.line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_image_ui.line_edit.textChanged.connect(lambda: self.on_source_image_ui_text_changed(source_image_ui))
        source_image_ui.line_edit.setPlaceholderText("Image Path")

        locate_source_action = actions.locate_source.create(self)
        locate_source_action.triggered.connect(
            lambda: self.thumbnail_viewer.select_image(source_image_ui.line_edit.text())
        )

        source_image_ui.context_menu = QMenu(self)
        source_image_ui.context_menu.aboutToShow.connect(
            lambda: self.set_override_metadata(self.get_source_image_metadata(source_image_ui))
        )
        source_image_ui.context_menu.aboutToHide.connect(
            lambda: QTimer.singleShot(0, lambda: self.set_override_metadata(None))
        )
        source_image_ui.context_menu.addAction(locate_source_action)
        source_image_ui.context_menu.addSeparator()
        source_image_ui.context_menu.addMenu(self.set_as_source_menu)

        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(2)
        vlayout.addStretch(1)
        vlayout.addWidget(source_label)
        vlayout.addWidget(source_image_ui.line_edit)
        vlayout.addStretch(2)

        hlayout = QHBoxLayout(source_image_ui.frame)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(2)
        hlayout.addLayout(vlayout)
        hlayout.addWidget(source_image_ui.label)

        return source_image_ui

    def remove_source_image_ui(self, source_image_ui: SourceImageUI):
        image_path = source_image_ui.line_edit.text()
        full_path = os.path.join(configuration.IMAGES_PATH, image_path)
        if os.path.exists(full_path):
            self.thumbnail_model.remove_reference(image_path)

    def on_source_image_ui_text_changed(self, source_image_ui: SourceImageUI):
        image_path = source_image_ui.line_edit.text()
        self.thumbnail_loader.get(
            image_path, 96, lambda image_path, pixmap: self.on_thumbnail_loaded(source_image_ui, pixmap)
        )

        # Update referenced images
        if source_image_ui.previous_image_path:
            self.thumbnail_model.remove_reference(source_image_ui.previous_image_path)
        self.thumbnail_model.add_reference(image_path)
        source_image_ui.previous_image_path = image_path

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

    def create_control_net_ui(self, condition_meta: ControlNetConditionMetadata) -> ControlNetConditionUI:
        condition_ui = ControlNetConditionUI()
        condition_ui.frame = FrameWithCloseButton()
        condition_ui.frame.closed.connect(lambda: self.on_remove_control_net(condition_ui))

        model_label = QLabel("Model")
        model_label.setAlignment(Qt.AlignCenter)
        condition_ui.model_combo_box = ComboBox()
        condition_ui.model_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        control_net_models = []
        if self.settings.value("install_control_net_v10", type=bool):
            control_net_models += control_net_config.v10_models
        if self.settings.value("install_control_net_v11", type=bool):
            control_net_models += control_net_config.v11_models
        if self.settings.value("install_control_net_mediapipe_v2", type=bool):
            control_net_models += control_net_config.mediapipe_v2_models
        condition_ui.model_combo_box.addItems(sorted(control_net_models))
        condition_ui.model_combo_box.setCurrentText(condition_meta.model)

        preprocessor_label = QLabel("Preprocessor")
        preprocessor_label.setAlignment(Qt.AlignCenter)
        condition_ui.preprocessor_combo_box = ComboBox()
        condition_ui.preprocessor_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        condition_ui.preprocessor_combo_box.addItems(control_net_config.preprocessors.keys())
        condition_ui.preprocessor_combo_box.currentTextChanged.connect(
            lambda text: self.on_control_net_preprocessor_combo_box_changed(condition_ui, text)
        )

        condition_ui.source_image_ui = self.create_source_image_ui()
        condition_ui.source_image_ui.line_edit.setText(condition_meta.source)

        sync_button = QPushButton(fa.icon_arrows_down_to_line)
        sync_button.setFont(fa.font)
        sync_button.setToolTip("Synchronize Model")
        sync_button.clicked.connect(lambda: self.on_control_net_sync_button_clicked(condition_ui))

        preview_preprocessor_button = QPushButton(fa.icon_eye)
        preview_preprocessor_button.setFont(fa.font)
        preview_preprocessor_button.setToolTip("Preview")
        preview_preprocessor_button.clicked.connect(
            lambda: self.on_control_net_preview_preprocessor_button_clicked(condition_ui)
        )

        condition_ui.scale = FloatSliderSpinBox("Scale", condition_meta.scale, maximum=2.0)

        model_vlayout = QVBoxLayout()
        model_vlayout.setContentsMargins(0, 0, 0, 0)
        model_vlayout.setSpacing(2)
        model_vlayout.addWidget(model_label)
        model_vlayout.addWidget(condition_ui.model_combo_box)

        preprocessor_hlayout = QHBoxLayout()
        preprocessor_hlayout.setContentsMargins(0, 0, 0, 0)
        preprocessor_hlayout.addWidget(condition_ui.preprocessor_combo_box)
        preprocessor_hlayout.addWidget(sync_button)
        preprocessor_hlayout.addWidget(preview_preprocessor_button)

        preprocessor_vlayout = QVBoxLayout()
        preprocessor_vlayout.setContentsMargins(0, 0, 0, 0)
        preprocessor_vlayout.setSpacing(2)
        preprocessor_vlayout.addWidget(preprocessor_label)
        preprocessor_vlayout.addLayout(preprocessor_hlayout)

        control_net_layout = QVBoxLayout(condition_ui.frame)
        control_net_layout.addLayout(preprocessor_vlayout)
        control_net_layout.addLayout(model_vlayout)
        control_net_layout.addWidget(condition_ui.source_image_ui.frame)
        condition_ui.params_layout_index = control_net_layout.count()
        control_net_layout.addWidget(condition_ui.scale)

        condition_ui.params = []
        condition_ui.preprocessor_combo_box.setCurrentText(condition_meta.preprocessor)
        self.set_control_net_param_values(condition_ui, condition_meta)

        return condition_ui

    def populate_set_as_source_menu(self):
        self.set_as_source_menu.clear()
        action = QAction("Image to Image", self)
        action.triggered.connect(lambda: self.on_set_as_source(self.img2img_source_ui))
        self.set_as_source_menu.addAction(action)

        for i, condition_ui in enumerate(self.condition_uis):
            action = QAction("Control Net {:d}".format(i + 1), self)
            action.triggered.connect(
                lambda checked=False, condition_ui=condition_ui: self.on_set_as_source(condition_ui.source_image_ui)
            )
            self.set_as_source_menu.addAction(action)

        self.set_as_source_menu.addSeparator()

        action = QAction("New Control Net Condition", self)
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

    def on_cancel_generation(self):
        if self.generate_thread:
            self.generate_thread.cancel = True

    def on_add_control_net(self) -> ControlNetConditionUI:
        condition_ui = self.add_control_net()
        self.update_config_frame_size()
        return condition_ui

    def on_remove_control_net(self, condition_ui: ControlNetConditionUI) -> None:
        self.remove_control_net(condition_ui)
        self.update_config_frame_size()

    def add_control_net(self) -> ControlNetConditionUI:
        condition_meta = ControlNetConditionMetadata()
        i = len(self.condition_uis)
        condition_ui = self.create_control_net_ui(condition_meta)
        self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, condition_ui.frame)
        self.condition_uis.append(condition_ui)
        return condition_ui

    def remove_control_net(self, condition_ui: ControlNetConditionUI) -> None:
        self.remove_source_image_ui(condition_ui.source_image_ui)
        index = self.control_net_group_box_layout.indexOf(condition_ui.frame) - self.control_net_dynamic_index
        del self.condition_uis[index]
        self.control_net_group_box_layout.removeWidget(condition_ui.frame)
        condition_ui.frame.setParent(None)

    def on_thumbnail_loaded(self, source_image_ui, pixmap):
        source_image_ui.label.setPixmap(pixmap)

    def get_control_net_param_values(self, condition_ui: ControlNetConditionUI) -> list[float]:
        result = []

        preprocessor = condition_ui.preprocessor_combo_box.currentText()
        preprocessor_type = control_net_config.preprocessors.get(preprocessor)
        if preprocessor_type:
            for param_ui in condition_ui.params:
                result.append(param_ui.value())

        return result

    def set_control_net_param_values(
        self, condition_ui: ControlNetConditionUI, condition_meta: ControlNetConditionMetadata
    ) -> None:
        preprocessor = condition_ui.preprocessor_combo_box.currentText()
        preprocessor_type = control_net_config.preprocessors.get(preprocessor)
        if preprocessor_type:
            for i, param_ui in enumerate(condition_ui.params):
                value = utils.list_get(condition_meta.params, i)
                if value:
                    param_ui.setValue(value)

    def on_control_net_preprocessor_combo_box_changed(self, condition_ui: ControlNetConditionUI, text: str):
        layout = condition_ui.frame.layout()
        for param in condition_ui.params:
            layout.removeWidget(param)
            param.setParent(None)

        condition_ui.params = []

        preprocessor_type = control_net_config.preprocessors.get(text)
        if preprocessor_type:
            for i, param in enumerate(preprocessor_type.params):
                if param.type == int:
                    param_ui = IntSliderSpinBox(param.name, param.value, param.min, param.max, param.step)
                elif param.type == float:
                    param_ui = FloatSliderSpinBox(param.name, param.value, param.min, param.max, param.step)
                else:
                    raise RuntimeError("Fatal error: Invalid type")

                layout.insertWidget(condition_ui.params_layout_index + i, param_ui)
                condition_ui.params.append(param_ui)

        self.update_config_frame_size()

    def on_control_net_sync_button_clicked(self, condition_ui: ControlNetConditionUI):
        preprocessor = condition_ui.preprocessor_combo_box.currentText()
        models = control_net_config.preprocessors_to_models.get(preprocessor, [])
        found = False
        for model in models:
            index = condition_ui.model_combo_box.findText(model)
            if index != -1:
                condition_ui.model_combo_box.setCurrentIndex(index)
                found = True
                break

        if not found:
            message_box = QMessageBox()
            message_box.setText("No Model Found")
            message_box.setInformativeText(
                "The '{:s}' preprocessor is not supported by any enabled ControlNet model.".format(preprocessor)
            )
            message_box.setIcon(QMessageBox.Warning)
            message_box.addButton(QMessageBox.Ok)
            message_box.exec()

    def on_control_net_preview_preprocessor_button_clicked(self, condition_ui: ControlNetConditionUI):
        source_path = condition_ui.source_image_ui.line_edit.text()
        width = self.width_spin_box.value()
        height = self.height_spin_box.value()

        full_path = os.path.join(configuration.IMAGES_PATH, source_path)
        with Image.open(full_path) as image:
            image = image.convert("RGB")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            source_image = image.copy()

        preprocessor_type = control_net_config.preprocessors[condition_ui.preprocessor_combo_box.currentText()]
        if preprocessor_type:
            params = self.get_control_net_param_values(condition_ui)

            if not isinstance(self.preview_preprocessor, preprocessor_type):
                # Use CPU to avoid race conditions with torch use on the generate thread
                self.preview_preprocessor = preprocessor_type(torch.device("cpu"))
            source_image = self.preview_preprocessor(source_image, params)
            if self.settings.value("reduce_memory", type=bool):
                self.preview_preprocessor = None
            output_path = "preprocessed.png"
            full_path = os.path.join(configuration.IMAGES_PATH, output_path)
            source_image.save(full_path)
            self.image_viewer.set_current_image(output_path)

    def on_generate_image(self):
        if self.generate_thread:
            return

        if not self.manual_seed_group_box.isChecked():
            self.randomize_seed()

        condition_metas = []
        for condition_ui in self.condition_uis:
            condition_meta = ControlNetConditionMetadata()
            condition_meta.model = condition_ui.model_combo_box.currentText()
            condition_meta.preprocessor = condition_ui.preprocessor_combo_box.currentText()
            condition_meta.source = condition_ui.source_image_ui.line_edit.text()
            condition_meta.params = self.get_control_net_param_values(condition_ui)
            condition_meta.scale = condition_ui.scale.value()
            condition_metas.append(condition_meta)

        self.settings.setValue("collection", self.thumbnail_viewer.collection())
        self.settings.setValue("model", self.model_combo_box.currentText())
        self.settings.setValue("scheduler", self.scheduler_combo_box.currentText())
        self.settings.setValue("prompt", self.prompt_edit.toPlainText())
        self.settings.setValue("negative_prompt", self.negative_prompt_edit.toPlainText())
        self.settings.setValue("manual_seed", self.manual_seed_group_box.isChecked())
        self.settings.setValue("seed", self.seed_line_edit.text())
        self.settings.setValue("num_images_per_prompt", self.num_images_spin_box.value())
        self.settings.setValue("num_inference_steps", self.num_steps_spin_box.value())
        self.settings.setValue("guidance_scale", self.guidance_scale_spin_box.value())
        self.settings.setValue("width", self.width_spin_box.value())
        self.settings.setValue("height", self.height_spin_box.value())
        self.settings.setValue("img2img_enabled", self.img2img_group_box.isChecked())
        self.settings.setValue("img2img_source", self.img2img_source_ui.line_edit.text())
        self.settings.setValue("img2img_noise", self.img2img_noise.value())
        self.settings.setValue("control_net_enabled", self.control_net_group_box.isChecked())
        self.settings.setValue("control_net_guidance_start", self.control_net_guidance_start.value())
        self.settings.setValue("control_net_guidance_end", self.control_net_guidance_end.value())
        self.settings.setValue(
            "control_net_conditions", json.dumps([condition_meta.to_dict() for condition_meta in condition_metas])
        )
        self.settings.setValue("upscale_enabled", self.upscale_group_box.isChecked())
        self.settings.setValue("upscale_factor", self.upscale_factor_combo_box.currentData())
        self.settings.setValue("upscale_denoising", self.upscale_denoising.value())
        self.settings.setValue("upscale_blend", self.upscale_blend.value())
        self.settings.setValue("face_enabled", self.face_restoration_group_box.isChecked())
        self.settings.setValue("face_blend", self.face_blend.value())
        self.settings.setValue("high_res_enabled", self.high_res_group_box.isChecked())
        self.settings.setValue("high_res_factor", self.high_res_factor.value())
        self.settings.setValue("high_res_steps", self.high_res_steps.value())
        self.settings.setValue("high_res_guidance_scale", self.high_res_guidance_scale.value())
        self.settings.setValue("high_res_noise", self.high_res_noise.value())

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
            self.progress_bar.setStyleSheet(
                "QProgressBar { border: none; } QProgressBar:chunk { background-color: grey; }"
            )
        else:
            self.progress_bar.setStyleSheet(
                "QProgressBar { border: none; } QProgressBar:chunk { background-color: blue; }"
            )
        if progress_amount is not None:
            self.progress_bar.setValue(progress_amount)
        else:
            self.progress_bar.setValue(0)

        if sys.platform == "darwin":
            sharedApplication = NSApplication.sharedApplication()
            dockTile = sharedApplication.dockTile()
            if maximum_amount == 0:
                dockTile.setBadgeLabel_("...")
            elif progress_amount is not None:
                dockTile.setBadgeLabel_("{:d}%".format(progress_amount))
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
        seed = random.randint(0, 0x7FFF_FFFF)
        self.seed_line_edit.setText(str(seed))

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
        self.prompt_edit.on_insert_textual_inversion()

    def on_insert_lora(self):
        self.prompt_edit.on_insert_lora()

    def on_interrogate(self):
        ci = Interrogator(
            Config(
                clip_model_name="ViT-L-14/openai",
                cache_path=os.path.join(configuration.MODELS_PATH, "interrogator"),
                device=torch.device,
            )
        )
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            full_path = os.path.join(configuration.IMAGES_PATH, image_metadata.path)
            with Image.open(full_path) as image:
                prompt = ci.interrogate(image)

        self.prompt_edit.setPlainText(prompt)

    def on_use_prompt(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.use_prompt(image_metadata)

    def on_use_seed(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.use_seed(image_metadata)

    def on_use_general(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.use_general(image_metadata)

    def on_use_source_images(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.use_source_images(image_metadata)
            self.update_config_frame_size()

    def on_use_img2img(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.use_img2img(image_metadata)

    def on_use_control_net(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.use_control_net(image_metadata)
            self.update_config_frame_size()

    def on_use_post_processing(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.use_post_processing(image_metadata)

    def on_use_all(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.use_prompt(image_metadata)
            self.use_seed(image_metadata)
            self.use_general(image_metadata)
            self.use_img2img(image_metadata)
            self.use_control_net(image_metadata)
            self.use_post_processing(image_metadata)
            self.use_source_images(image_metadata)
            self.update_config_frame_size()

    def use_prompt(self, image_metadata: ImageMetadata):
        self.prompt_edit.setPlainText(image_metadata.prompt)
        self.negative_prompt_edit.setPlainText(image_metadata.negative_prompt)

    def use_seed(self, image_metadata: ImageMetadata):
        self.manual_seed_group_box.setChecked(True)
        self.seed_line_edit.setText(str(image_metadata.seed))

    def use_general(self, image_metadata: ImageMetadata):
        self.num_steps_spin_box.setValue(image_metadata.num_inference_steps)
        self.guidance_scale_spin_box.setValue(image_metadata.guidance_scale)
        self.width_spin_box.setValue(image_metadata.width)
        self.height_spin_box.setValue(image_metadata.height)
        self.scheduler_combo_box.setCurrentText(image_metadata.scheduler)

    def use_source_images(self, image_metadata: ImageMetadata):
        img2img_meta = image_metadata.img2img
        if img2img_meta:
            self.img2img_source_ui.line_edit.setText(img2img_meta.source)

        control_net_meta = image_metadata.control_net
        if control_net_meta:
            while len(self.condition_uis) < len(control_net_meta.conditions):
                self.add_control_net()

            for i, condition_meta in enumerate(control_net_meta.conditions):
                condition_ui = self.condition_uis[i]
                condition_ui.source_image_ui.line_edit.setText(condition_meta.source)

    def use_img2img(self, image_metadata: ImageMetadata):
        img2img_meta = image_metadata.img2img
        if img2img_meta:
            self.img2img_group_box.setChecked(True)
            self.img2img_source_ui.line_edit.setText(img2img_meta.source)
            self.img2img_noise.setValue(img2img_meta.noise)
        else:
            self.img2img_group_box.setChecked(False)
            self.img2img_source_ui.line_edit.setText("")

    def use_control_net(self, image_metadata: ImageMetadata):
        for condition_ui in self.condition_uis.copy():
            self.remove_control_net(condition_ui)

        self.condition_uis = []

        control_net_meta = image_metadata.control_net
        if control_net_meta:
            self.control_net_group_box.setChecked(True)
            self.control_net_guidance_start.setValue(control_net_meta.guidance_start)
            self.control_net_guidance_end.setValue(control_net_meta.guidance_end)

            for i, condition_meta in enumerate(control_net_meta.conditions):
                condition_ui = self.create_control_net_ui(condition_meta)
                self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, condition_ui.frame)
                self.condition_uis.append(condition_ui)
        else:
            self.control_net_group_box.setChecked(False)

    def use_post_processing(self, image_metadata: ImageMetadata):
        upscale_meta = image_metadata.upscale
        if upscale_meta:
            self.upscale_group_box.setChecked(True)
            utils.set_current_data(self.upscale_factor_combo_box, upscale_meta.factor)
            self.upscale_denoising.setValue(upscale_meta.denoising)
            self.upscale_blend.setValue(upscale_meta.blend)
        else:
            self.upscale_group_box.setChecked(False)

        face_meta = image_metadata.face
        if face_meta:
            self.face_restoration_group_box.setChecked(True)
            self.face_blend.setValue(face_meta.blend)
        else:
            self.face_restoration_group_box.setChecked(False)

        high_res_meta = image_metadata.high_res
        if high_res_meta:
            self.high_res_group_box.setChecked(True)
            self.high_res_factor.setValue(high_res_meta.factor)
            self.high_res_steps.setValue(high_res_meta.steps)
            self.high_res_guidance_scale.setValue(high_res_meta.guidance_scale)
            self.high_res_noise.setValue(high_res_meta.noise)
        else:
            self.high_res_group_box.setChecked(False)

    def on_move_image(self, collection: str):
        current_collection = self.thumbnail_viewer.collection()
        if collection == current_collection:
            return

        image_metadata = self.get_current_metadata()
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

    def on_close(self):
        if self.generate_thread:
            self.generate_thread.cancel = True
            self.generate_thread.finished.connect(self.quit)
            return False
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
