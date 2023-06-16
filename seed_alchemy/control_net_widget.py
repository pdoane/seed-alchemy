from PySide6.QtCore import QPoint, QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import control_net_config
from . import font_awesome as fa
from . import utils
from .image_metadata import ControlNetConditionMetadata
from .source_image_widget import SourceImageWidget
from .thumbnail_loader import ThumbnailLoader
from .thumbnail_model import ThumbnailModel
from .widgets import (
    ComboBox,
    FloatSliderSpinBox,
    FrameWithCloseButton,
    IntSliderSpinBox,
)


class ControlNetWidget(QWidget):
    closed = Signal()
    preview_clicked = Signal()
    source_image_context_menu_requested = Signal(SourceImageWidget, QPoint)
    layout_changed = Signal()

    def __init__(
        self,
        settings: QSettings,
        thumbnail_loader: ThumbnailLoader,
        thumbnail_model: ThumbnailModel,
        condition_meta: ControlNetConditionMetadata,
        parent=None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.thumbnail_loader = thumbnail_loader
        self.thumbnail_model = thumbnail_model

        self.frame: FrameWithCloseButton = None
        self.model_combo_box: ComboBox = None
        self.preprocessor_combo_box: ComboBox = None
        self.source_image_widget: SourceImageWidget = None
        self.params_layout_index: int = 0
        self.params: list[QWidget] = []
        self.scale: FloatSliderSpinBox = None
        self.init_ui(condition_meta)

    def init_ui(self, condition_meta: ControlNetConditionMetadata):
        self.frame = FrameWithCloseButton()
        self.frame.closed.connect(self.on_frame_closed)

        model_label = QLabel("Model")
        model_label.setAlignment(Qt.AlignCenter)
        self.model_combo_box = ComboBox()
        self.model_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        control_net_models = []
        if self.settings.value("install_control_net_v10", type=bool):
            control_net_models += control_net_config.v10_models
        if self.settings.value("install_control_net_v11", type=bool):
            control_net_models += control_net_config.v11_models
        if self.settings.value("install_control_net_mediapipe_v2", type=bool):
            control_net_models += control_net_config.mediapipe_v2_models
        self.model_combo_box.addItems(sorted(control_net_models))
        self.model_combo_box.setCurrentText(condition_meta.model)

        preprocessor_label = QLabel("Preprocessor")
        preprocessor_label.setAlignment(Qt.AlignCenter)
        self.preprocessor_combo_box = ComboBox()
        self.preprocessor_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.preprocessor_combo_box.addItems(control_net_config.preprocessors.keys())
        self.preprocessor_combo_box.currentTextChanged.connect(self.on_preprocessor_combo_box_changed)

        self.source_image_widget = SourceImageWidget(self.thumbnail_loader, self.thumbnail_model)
        self.source_image_widget.label.customContextMenuRequested.connect(self.on_source_image_context_menu_requested)
        self.source_image_widget.line_edit.setText(condition_meta.source)

        sync_button = QPushButton(fa.icon_arrows_down_to_line)
        sync_button.setFont(fa.font)
        sync_button.setToolTip("Synchronize Model")
        sync_button.clicked.connect(self.on_sync_button_clicked)

        preview_button = QPushButton(fa.icon_eye)
        preview_button.setFont(fa.font)
        preview_button.setToolTip("Preview")
        preview_button.clicked.connect(self.on_preview_button_clicked)

        self.scale = FloatSliderSpinBox("Scale", maximum=2.0)
        self.scale.setValue(condition_meta.scale)

        model_vlayout = QVBoxLayout()
        model_vlayout.setContentsMargins(0, 0, 0, 0)
        model_vlayout.setSpacing(2)
        model_vlayout.addWidget(model_label)
        model_vlayout.addWidget(self.model_combo_box)

        preprocessor_hlayout = QHBoxLayout()
        preprocessor_hlayout.setContentsMargins(0, 0, 0, 0)
        preprocessor_hlayout.addWidget(self.preprocessor_combo_box)
        preprocessor_hlayout.addWidget(sync_button)
        preprocessor_hlayout.addWidget(preview_button)

        preprocessor_vlayout = QVBoxLayout()
        preprocessor_vlayout.setContentsMargins(0, 0, 0, 0)
        preprocessor_vlayout.setSpacing(2)
        preprocessor_vlayout.addWidget(preprocessor_label)
        preprocessor_vlayout.addLayout(preprocessor_hlayout)

        control_net_layout = QVBoxLayout(self.frame)
        control_net_layout.addLayout(preprocessor_vlayout)
        control_net_layout.addLayout(model_vlayout)
        control_net_layout.addWidget(self.source_image_widget)
        self.params_layout_index = control_net_layout.count()
        control_net_layout.addWidget(self.scale)

        self.params = []
        self.preprocessor_combo_box.setCurrentText(condition_meta.preprocessor)
        self.set_param_values(condition_meta)

    def get_condition_meta(self):
        condition_meta = ControlNetConditionMetadata()
        condition_meta.model = self.get_model_name()
        condition_meta.preprocessor = self.get_preprocessor_name()
        condition_meta.source = self.get_source_path()
        condition_meta.params = self.get_param_values()
        condition_meta.scale = self.get_scale()
        return condition_meta

    def get_model_name(self):
        return self.model_combo_box.currentText()

    def get_preprocessor_name(self):
        return self.preprocessor_combo_box.currentText()

    def get_source_path(self):
        return self.source_image_widget.get_image_path()

    def get_scale(self):
        return self.scale.value()

    def get_param_values(self) -> list[float]:
        result = []

        preprocessor = self.preprocessor_combo_box.currentText()
        preprocessor_type = control_net_config.preprocessors.get(preprocessor)
        if preprocessor_type:
            for param_ui in self.params:
                result.append(param_ui.value())

        return result

    def set_param_values(self, condition_meta: ControlNetConditionMetadata) -> None:
        preprocessor = self.preprocessor_combo_box.currentText()
        preprocessor_type = control_net_config.preprocessors.get(preprocessor)
        if preprocessor_type:
            for i, param_ui in enumerate(self.params):
                value = utils.list_get(condition_meta.params, i)
                if value:
                    param_ui.setValue(value)

    def on_frame_closed(self):
        self.closed.emit()

    def on_preprocessor_combo_box_changed(self, text: str):
        layout = self.frame.layout()
        for param in self.params:
            layout.removeWidget(param)
            param.setParent(None)

        self.params = []

        preprocessor_type = control_net_config.preprocessors.get(text)
        if preprocessor_type:
            for i, param in enumerate(preprocessor_type.params):
                if param.type == int:
                    param_ui = IntSliderSpinBox(param.name, param.min, param.max, param.step)
                elif param.type == float:
                    param_ui = FloatSliderSpinBox(param.name, param.min, param.max, param.step)
                else:
                    raise RuntimeError("Fatal error: Invalid type")
                param_ui.setValue(param.value)

                layout.insertWidget(self.params_layout_index + i, param_ui)
                self.params.append(param_ui)

        self.layout_changed.emit()

    def on_sync_button_clicked(self):
        preprocessor = self.preprocessor_combo_box.currentText()
        models = control_net_config.preprocessors_to_models.get(preprocessor, [])
        found = False
        for model in models:
            index = self.model_combo_box.findText(model)
            if index != -1:
                self.model_combo_box.setCurrentIndex(index)
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

    def on_preview_button_clicked(self):
        self.preview_clicked.emit()

    def on_source_image_context_menu_requested(self, point):
        self.source_image_context_menu_requested.emit(self.source_image_widget, point)
