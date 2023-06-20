import random

from PySide6.QtCore import QSettings, QSize, Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import actions, configuration, scheduler_registry
from . import font_awesome as fa
from . import utils
from .control_net_widget import ControlNetWidget
from .icon_engine import FontAwesomeIconEngine
from .image_metadata import (
    ControlNetConditionMetadata,
    ControlNetMetadata,
    FaceRestorationMetadata,
    HighResMetadata,
    ImageMetadata,
    Img2ImgMetadata,
    InpaintMetadata,
    UpscaleMetadata,
)
from .prompt_text_edit import PromptTextEdit
from .source_image_widget import SourceImageWidget
from .thumbnail_loader import ThumbnailLoader
from .thumbnail_model import ThumbnailModel
from .widgets import (
    ComboBox,
    DoubleSpinBox,
    FloatSliderSpinBox,
    IntSliderSpinBox,
    ScrollArea,
    SpinBox,
)


class ImageGenerationPanel(QWidget):
    image_size_changed = Signal(QSize)
    generate_requested = Signal()
    cancel_requested = Signal()
    preprocess_requested = Signal(ControlNetWidget)
    locate_source_requested = Signal(str)

    def __init__(self, main_window, mode: str, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.settings: QSettings = main_window.settings
        self.thumbnail_loader: ThumbnailLoader = main_window.thumbnail_loader
        self.thumbnail_model: ThumbnailModel = main_window.thumbnail_model
        self.mode = mode

        self.needs_fix = False

        # Generate
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.on_generate_clicked)

        cancel_button = actions.cancel_generation.push_button()
        cancel_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cancel_button.clicked.connect(self.on_cancel_clicked)

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

        # Prompts
        self.prompt_edit = PromptTextEdit(8, "Prompt")
        self.prompt_edit.return_pressed.connect(self.on_generate_clicked)
        self.negative_prompt_edit = PromptTextEdit(5, "Negative Prompt")
        self.negative_prompt_edit.return_pressed.connect(self.on_generate_clicked)

        self.prompt_group_box = QGroupBox("Prompts")
        prompt_group_box_layout = QVBoxLayout(self.prompt_group_box)
        prompt_group_box_layout.addWidget(self.prompt_edit)
        prompt_group_box_layout.addWidget(self.negative_prompt_edit)

        # General
        model_label = QLabel("Stable Diffusion Model")
        model_label.setAlignment(Qt.AlignCenter)
        self.model_combo_box = ComboBox()
        self.model_combo_box.addItems(configuration.stable_diffusion_models.keys())

        scheduler_label = QLabel("Scheduler")
        scheduler_label.setAlignment(Qt.AlignCenter)
        self.scheduler_combo_box = ComboBox()
        self.scheduler_combo_box.addItems(scheduler_registry.DICT.keys())

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
        self.width_spin_box.valueChanged.connect(self.on_image_size_changed)
        height_label = QLabel("Height")
        height_label.setAlignment(Qt.AlignCenter)
        self.height_spin_box = SpinBox()
        self.height_spin_box.setAlignment(Qt.AlignCenter)
        self.height_spin_box.setFixedWidth(80)
        self.height_spin_box.setSingleStep(64)
        self.height_spin_box.setMinimum(64)
        self.height_spin_box.setMaximum(2048)
        self.height_spin_box.valueChanged.connect(self.on_image_size_changed)

        self.general_group_box = QGroupBox("General")

        model_vlayout = QVBoxLayout()
        model_vlayout.setContentsMargins(0, 0, 0, 0)
        model_vlayout.setSpacing(2)
        model_vlayout.addWidget(model_label)
        model_vlayout.addWidget(self.model_combo_box)

        scheduler_vlayout = QVBoxLayout()
        scheduler_vlayout.setContentsMargins(0, 0, 0, 0)
        scheduler_vlayout.setSpacing(2)
        scheduler_vlayout.addWidget(scheduler_label)
        scheduler_vlayout.addWidget(self.scheduler_combo_box)

        controls_grid = QGridLayout()
        controls_grid.setVerticalSpacing(2)
        controls_grid.setRowMinimumHeight(2, 10)
        controls_grid.addWidget(num_images_label, 0, 0)
        controls_grid.addWidget(self.num_images_spin_box, 1, 0)
        controls_grid.addWidget(num_steps_label, 0, 1)
        controls_grid.addWidget(self.num_steps_spin_box, 1, 1)
        controls_grid.addWidget(guidance_scale_label, 0, 2)
        controls_grid.addWidget(self.guidance_scale_spin_box, 1, 2)
        controls_grid.addWidget(width_label, 3, 0)
        controls_grid.addWidget(self.width_spin_box, 4, 0)
        controls_grid.addWidget(height_label, 3, 1)
        controls_grid.addWidget(self.height_spin_box, 4, 1)

        controls_vlayout = QVBoxLayout(self.general_group_box)
        controls_vlayout.addLayout(model_vlayout)
        controls_vlayout.addLayout(scheduler_vlayout)
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
        if self.mode == "image":
            self.img2img_source_widget = self.create_source_image_widget()
            self.img2img_noise = FloatSliderSpinBox("Noise")

            self.img2img_group_box = QGroupBox("Image to Image")
            self.img2img_group_box.setCheckable(True)

            img2img_group_box_layout = QVBoxLayout(self.img2img_group_box)
            img2img_group_box_layout.addWidget(self.img2img_source_widget)
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

        # Inpaint
        if self.mode == "image":
            self.inpaint_source_widget = self.create_source_image_widget()
            self.inpaint_use_alpha_channel_check_box = QCheckBox("Use Alpha")
            self.inpaint_invert_mask_check_box = QCheckBox("Invert Mask")

            self.inpaint_group_box = QGroupBox("Inpainting")
            self.inpaint_group_box.setCheckable(True)

            inpaint_hlayout = QHBoxLayout()
            inpaint_hlayout.addWidget(self.inpaint_use_alpha_channel_check_box)
            inpaint_hlayout.addWidget(self.inpaint_invert_mask_check_box)

            inpaint_group_box_layout = QVBoxLayout(self.inpaint_group_box)
            inpaint_group_box_layout.addWidget(self.inpaint_source_widget)
            inpaint_group_box_layout.addLayout(inpaint_hlayout)
        elif self.mode == "canvas":
            self.inpaint_noise = FloatSliderSpinBox("Noise")
            self.inpaint_group_box = QGroupBox("Inpainting")

            inpaint_group_box_layout = QVBoxLayout(self.inpaint_group_box)
            inpaint_group_box_layout.addWidget(self.inpaint_noise)

        # Configuration
        config_layout = QVBoxLayout(self.config_frame)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self.prompt_group_box)
        config_layout.addWidget(self.general_group_box)
        config_layout.addWidget(self.manual_seed_group_box)
        if self.mode == "image":
            config_layout.addWidget(self.img2img_group_box)
        else:
            config_layout.addWidget(self.inpaint_group_box)
        config_layout.addWidget(self.control_net_group_box)
        config_layout.addWidget(self.upscale_group_box)
        config_layout.addWidget(self.face_restoration_group_box)
        config_layout.addWidget(self.high_res_group_box)
        if self.mode == "image":
            config_layout.addWidget(self.inpaint_group_box)
        config_layout.addStretch()

        panel_layout = QVBoxLayout(self)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addLayout(generate_hlayout)
        panel_layout.addWidget(self.config_scroll_area)

    def serialize(self) -> dict:
        base_dict = {
            "mode": self.mode,
            "model": self.model_combo_box.currentText(),
            "safety_checker": self.settings.value("safety_checker", type=bool),
            "scheduler": self.scheduler_combo_box.currentText(),
            "prompt": self.prompt_edit.toPlainText(),
            "negative_prompt": self.negative_prompt_edit.toPlainText(),
            "manual_seed": self.manual_seed_group_box.isChecked(),
            "seed": int(self.seed_line_edit.text()),
            "num_images_per_prompt": self.num_images_spin_box.value(),
            "num_inference_steps": self.num_steps_spin_box.value(),
            "guidance_scale": self.guidance_scale_spin_box.value(),
            "width": self.width_spin_box.value(),
            "height": self.height_spin_box.value(),
            "control_net_enabled": self.control_net_group_box.isChecked(),
            "control_net_guidance_start": self.control_net_guidance_start.value(),
            "control_net_guidance_end": self.control_net_guidance_end.value(),
            "control_net_conditions": [
                control_net_widget.get_condition_meta().to_dict() for control_net_widget in self.control_net_widgets
            ],
            "upscale_enabled": self.upscale_group_box.isChecked(),
            "upscale_factor": self.upscale_factor_combo_box.currentData(),
            "upscale_denoising": self.upscale_denoising.value(),
            "upscale_blend": self.upscale_blend.value(),
            "face_enabled": self.face_restoration_group_box.isChecked(),
            "face_blend": self.face_blend.value(),
            "high_res_enabled": self.high_res_group_box.isChecked(),
            "high_res_factor": self.high_res_factor.value(),
            "high_res_steps": self.high_res_steps.value(),
            "high_res_guidance_scale": self.high_res_guidance_scale.value(),
            "high_res_noise": self.high_res_noise.value(),
        }

        if self.mode == "image":
            image_dict = {
                "img2img_enabled": self.img2img_group_box.isChecked(),
                "img2img_source": self.img2img_source_widget.line_edit.text(),
                "img2img_noise": self.img2img_noise.value(),
                "inpaint_enabled": self.inpaint_group_box.isChecked(),
                "inpaint_source": self.inpaint_source_widget.get_image_path(),
                "inpaint_use_alpha_channel": self.inpaint_use_alpha_channel_check_box.isChecked(),
                "inpaint_invert_mask": self.inpaint_invert_mask_check_box.isChecked(),
            }
            return {**base_dict, **image_dict}
        elif self.mode == "canvas":
            canvas_dict = {
                "inpaint_noise": self.inpaint_noise.value(),
            }
            return {**base_dict, **canvas_dict}

    def deserialize(self, data):
        image_meta = ImageMetadata()
        img2img_meta = Img2ImgMetadata()
        control_net_meta = ControlNetMetadata()
        upscale_meta = UpscaleMetadata()
        face_meta = FaceRestorationMetadata()
        high_res_meta = HighResMetadata()
        inpaint_meta = InpaintMetadata()

        self.model_combo_box.setCurrentText(data.get("model", image_meta.model))
        self.prompt_edit.setPlainText(data.get("prompt", image_meta.prompt))
        self.negative_prompt_edit.setPlainText(data.get("negative_prompt", image_meta.negative_prompt))
        self.num_images_spin_box.setValue(data.get("num_images_per_prompt", 1))
        self.num_steps_spin_box.setValue(data.get("num_inference_steps", image_meta.num_inference_steps))
        self.guidance_scale_spin_box.setValue(data.get("guidance_scale", image_meta.guidance_scale))
        self.width_spin_box.setValue(data.get("width", image_meta.width))
        self.height_spin_box.setValue(data.get("height", image_meta.height))
        self.scheduler_combo_box.setCurrentText(data.get("scheduler", image_meta.scheduler))
        self.seed_line_edit.setText(str(data.get("seed", image_meta.seed)))
        self.manual_seed_group_box.setChecked(data.get("manual_seed", False))
        if self.mode == "image":
            self.img2img_source_widget.line_edit.setText(data.get("img2img_source", img2img_meta.source))
            self.img2img_noise.setValue(data.get("img2img_noise", img2img_meta.noise))
            self.img2img_group_box.setChecked(data.get("img2img_enabled", False))
        self.control_net_guidance_start.setValue(
            data.get("control_net_guidance_start", control_net_meta.guidance_start)
        )
        self.control_net_guidance_end.setValue(data.get("control_net_guidance_end", control_net_meta.guidance_end))
        self.control_net_group_box.setChecked(data.get("control_net_enabled", False))
        self.control_net_widgets = []
        for i, condition_meta_data in enumerate(data.get("control_net_conditions", [])):
            condition_meta = ControlNetConditionMetadata.from_dict(condition_meta_data)
            control_net_widget = self.create_control_net_widget(condition_meta)
            self.control_net_group_box_layout.insertWidget(
                self.control_net_dynamic_index + i, control_net_widget.frame
            )
            self.control_net_widgets.append(control_net_widget)
        utils.set_current_data(self.upscale_factor_combo_box, data.get("upscale_factor", upscale_meta.factor))
        self.upscale_denoising.setValue(data.get("upscale_denoising", upscale_meta.denoising))
        self.upscale_blend.setValue(data.get("upscale_blend", upscale_meta.blend))
        self.upscale_group_box.setChecked(data.get("upscale_enabled", False))
        self.face_blend.setValue(data.get("face_blend", face_meta.blend))
        self.face_restoration_group_box.setChecked(data.get("face_enabled", False))
        self.high_res_factor.setValue(data.get("high_res_factor", high_res_meta.factor))
        self.high_res_steps.setValue(data.get("high_res_steps", high_res_meta.steps))
        self.high_res_guidance_scale.setValue(data.get("high_res_guidance_scale", high_res_meta.guidance_scale))
        self.high_res_noise.setValue(data.get("high_res_noise", high_res_meta.noise))
        self.high_res_group_box.setChecked(data.get("high_res_enabled", False))
        if self.mode == "image":
            self.inpaint_source_widget.line_edit.setText(data.get("inpaint_source", inpaint_meta.source))
            self.inpaint_use_alpha_channel_check_box.setChecked(
                data.get("inpaint_use_alpha_channel", inpaint_meta.use_alpha_channel)
            )
            self.inpaint_invert_mask_check_box.setChecked(data.get("inpaint_invert_mask", inpaint_meta.invert_mask))
            self.inpaint_group_box.setChecked(data.get("inpaint_enabled", False))
        elif self.mode == "canvas":
            self.inpaint_noise.setValue(data.get("inpaint_noise", img2img_meta.noise))

    def get_image_size(self):
        width = self.width_spin_box.value()
        height = self.height_spin_box.value()
        return QSize(width, height)

    def set_image_size(self, image_size):
        self.width_spin_box.setValue(image_size.width())
        self.height_spin_box.setValue(image_size.height())

    def update_config_frame_size(self):
        self.config_frame.adjustSize()

        # Workaround QT issue where the size of QScrollArea.widget() is cached and not updated
        scroll_pos = self.config_scroll_area.verticalScrollBar().value()
        self.config_scroll_area.takeWidget()
        self.config_scroll_area.setWidget(self.config_frame)
        self.config_scroll_area.updateGeometry()
        self.config_scroll_area.adjustSize()
        self.config_scroll_area.verticalScrollBar().setValue(scroll_pos)

    def create_source_image_widget(self) -> SourceImageWidget:
        source_image_widget = SourceImageWidget(self.thumbnail_loader, self.thumbnail_model)
        source_image_widget.label.customContextMenuRequested.connect(
            lambda point: self.show_source_image_context_menu(source_image_widget, point)
        )
        return source_image_widget

    def show_source_image_context_menu(self, source_image_widget: SourceImageWidget, point):
        image_metadata = source_image_widget.get_metadata()

        locate_source_action = actions.locate_source.create(self)
        locate_source_action.triggered.connect(lambda: self.on_locate_source(image_metadata))

        set_as_source_menu = QMenu("Set as Source", self)
        set_as_source_menu.setIcon(QIcon(FontAwesomeIconEngine(fa.icon_share)))
        self.build_set_as_source_menu(set_as_source_menu, image_metadata)

        context_menu = QMenu(self)
        context_menu.addAction(locate_source_action)
        context_menu.addSeparator()
        context_menu.addMenu(set_as_source_menu)

        context_menu.exec(source_image_widget.label.mapToGlobal(point))

    def create_control_net_widget(self, condition_meta: ControlNetConditionMetadata) -> ControlNetWidget:
        control_net_widget = ControlNetWidget(self.thumbnail_loader, self.thumbnail_model, condition_meta)
        control_net_widget.closed.connect(lambda: self.on_remove_control_net(control_net_widget))
        control_net_widget.preview_clicked.connect(lambda: self.on_preview_clicked(control_net_widget))
        control_net_widget.source_image_context_menu_requested.connect(self.show_source_image_context_menu)
        control_net_widget.layout_changed.connect(self.update_config_frame_size)
        return control_net_widget

    def build_set_as_source_menu(self, menu: QMenu, image_metadata: ImageMetadata):
        action = QAction("Image to Image", self)
        action.triggered.connect(lambda: self.on_set_as_source(self.img2img_source_widget, image_metadata))
        menu.addAction(action)

        for i, control_net_widget in enumerate(self.control_net_widgets):
            source_image_widget = control_net_widget.source_image_widget
            action = QAction("Control Net {:d}".format(i + 1), self)
            action.triggered.connect(
                lambda checked=False, source_image_widget=source_image_widget: self.on_set_as_source(
                    source_image_widget, image_metadata
                )
            )
            menu.addAction(action)

        action = QAction("Control Net (New Condition)", self)
        action.triggered.connect(
            lambda: self.on_set_as_source(self.on_add_control_net().source_image_widget, image_metadata)
        )
        menu.addAction(action)

        menu.addSeparator()

        action = QAction("Canvas", self)
        action.triggered.connect(lambda: self.on_canvas_mode(image_metadata.path))
        menu.addAction(action)

        action = QAction("Interrogate", self)
        action.triggered.connect(lambda: self.on_interrogate_mode(image_metadata.path))
        menu.addAction(action)

    def on_locate_source(self, image_metadata: ImageMetadata):
        self.locate_source_requested.emit(image_metadata.path)

    def on_set_as_source(self, source_image_widget: SourceImageWidget, image_metadata: ImageMetadata):
        if image_metadata is not None:
            source_image_widget.line_edit.setText(image_metadata.path)
            if source_image_widget == self.img2img_source_widget:
                self.img2img_group_box.setChecked(True)
                self.width_spin_box.setValue(image_metadata.width)
                self.height_spin_box.setValue(image_metadata.height)
            else:
                self.control_net_group_box.setChecked(True)

    def on_canvas_mode(self, image_path):
        canvas_mode_widget = self.main_window.set_mode("canvas")
        canvas_mode_widget.add_image(image_path)

    def on_interrogate_mode(self, image_path):
        interrogate_mode_widget = self.main_window.set_mode("interrogate")
        interrogate_mode_widget.source_image_widget.line_edit.setText(image_path)

    def on_image_size_changed(self):
        self.image_size_changed.emit(self.get_image_size())

    def on_cancel_clicked(self):
        self.cancel_requested.emit()

    def on_add_control_net(self) -> ControlNetWidget:
        control_net_widget = self.add_control_net()
        self.update_config_frame_size()
        return control_net_widget

    def on_remove_control_net(self, control_net_widget: ControlNetWidget) -> None:
        self.remove_control_net(control_net_widget)
        self.update_config_frame_size()

    def add_control_net(self) -> ControlNetWidget:
        condition_meta = ControlNetConditionMetadata()
        i = len(self.control_net_widgets)
        control_net_widget = self.create_control_net_widget(condition_meta)
        self.control_net_group_box_layout.insertWidget(self.control_net_dynamic_index + i, control_net_widget.frame)
        self.control_net_widgets.append(control_net_widget)
        return control_net_widget

    def remove_control_net(self, control_net_widget: ControlNetWidget) -> None:
        control_net_widget.source_image_widget.destroy()
        index = self.control_net_group_box_layout.indexOf(control_net_widget.frame) - self.control_net_dynamic_index
        del self.control_net_widgets[index]
        self.control_net_group_box_layout.removeWidget(control_net_widget.frame)
        control_net_widget.frame.setParent(None)

    def on_preview_clicked(self, control_net_widget: ControlNetWidget):
        self.preprocess_requested.emit(control_net_widget)

    def on_generate_clicked(self):
        if not self.manual_seed_group_box.isChecked():
            self.randomize_seed()

        self.generate_requested.emit()

    def randomize_seed(self):
        seed = random.randint(0, 0x7FFF_FFFF)
        self.seed_line_edit.setText(str(seed))

    def on_seed_random_clicked(self):
        self.randomize_seed()

    def begin_generate(self):
        self.generate_button.setEnabled(False)

    def end_generate(self):
        self.generate_button.setEnabled(True)

    def begin_update(self):
        pass

    def end_update(self):
        if self.needs_fix:
            self.update_config_frame_size()
            self.needs_fix = False

    def use_all(self, image_metadata: ImageMetadata):
        self.use_prompt(image_metadata)
        self.use_seed(image_metadata)
        self.use_general(image_metadata)
        self.use_img2img(image_metadata)
        self.use_control_net(image_metadata)
        self.use_post_processing(image_metadata)
        self.use_source_images(image_metadata)
        self.use_inpaint(image_metadata)

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
            self.img2img_source_widget.line_edit.setText(img2img_meta.source)

        control_net_meta = image_metadata.control_net
        if control_net_meta:
            while len(self.control_net_widgets) < len(control_net_meta.conditions):
                self.add_control_net()

            for i, condition_meta in enumerate(control_net_meta.conditions):
                control_net_widget = self.control_net_widgets[i]
                control_net_widget.source_image_widget.line_edit.setText(condition_meta.source)

        self.needs_fix = True

    def use_img2img(self, image_metadata: ImageMetadata):
        if self.mode == "image":
            img2img_meta = image_metadata.img2img
            if img2img_meta:
                self.img2img_group_box.setChecked(True)
                self.img2img_source_widget.line_edit.setText(img2img_meta.source)
                self.img2img_noise.setValue(img2img_meta.noise)
            else:
                self.img2img_group_box.setChecked(False)
                self.img2img_source_widget.line_edit.setText("")

    def use_control_net(self, image_metadata: ImageMetadata):
        for control_net_widget in self.control_net_widgets.copy():
            self.remove_control_net(control_net_widget)

        self.control_net_widgets = []

        control_net_meta = image_metadata.control_net
        if control_net_meta:
            self.control_net_group_box.setChecked(True)
            self.control_net_guidance_start.setValue(control_net_meta.guidance_start)
            self.control_net_guidance_end.setValue(control_net_meta.guidance_end)

            for i, condition_meta in enumerate(control_net_meta.conditions):
                control_net_widget = self.create_control_net_widget(condition_meta)
                self.control_net_group_box_layout.insertWidget(
                    self.control_net_dynamic_index + i, control_net_widget.frame
                )
                self.control_net_widgets.append(control_net_widget)
        else:
            self.control_net_group_box.setChecked(False)

        self.needs_fix = True

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

    def use_inpaint(self, image_metadata: ImageMetadata):
        if self.mode == "image":
            inpaint_meta = image_metadata.inpaint
            if inpaint_meta:
                self.inpaint_group_box.setChecked(True)
                self.img2img_source_widget.line_edit.setText(inpaint_meta.source)
                self.inpaint_use_alpha_channel_check_box.setChecked(inpaint_meta.use_alpha_channel)
                self.inpaint_invert_mask_check_box.setChecked(inpaint_meta.invert_mask)
            else:
                self.inpaint_group_box.setChecked(False)
        elif self.mode == "canvas":
            img2img_meta = image_metadata.img2img
            if img2img_meta:
                self.inpaint_noise.setValue(img2img_meta.noise)
