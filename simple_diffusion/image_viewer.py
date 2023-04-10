import os

import actions
import configuration
import utils
from configuration import ControlNetCondition
from image_metadata import ImageMetadata
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QPixmap
from PySide6.QtWidgets import (QApplication, QFrame,
                               QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget)

class MetadataRow:
    def __init__(self, label_text, multiline=False):
        self.label = QLabel(label_text)
        self.label.setStyleSheet('font-weight: bold; background-color: transparent')

        self.value = QLabel()
        self.value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.value.setStyleSheet('background-color: transparent')
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.value.setCursor(Qt.IBeamCursor)

        if multiline:
            self.value.setWordWrap(True)

        self.frame = QFrame()
        self.frame.setContentsMargins(0, 0, 0, 0)
        self.frame.setStyleSheet('background-color: transparent')

        hlayout = QHBoxLayout(self.frame)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(self.label)
        hlayout.addWidget(self.value)

class ImageMetadataFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet('background-color: rgba(0, 0, 0, 127);')

        self.path = MetadataRow('Path:')
        self.type = MetadataRow('Type:')
        self.model = MetadataRow('Model:')
        self.scheduler = MetadataRow('Scheduler:')
        self.prompt = MetadataRow('Prompt:', multiline=True)
        self.negative_prompt = MetadataRow('Negative Prompt:', multiline=True)
        self.seed = MetadataRow('Seed:')
        self.num_inference_steps = MetadataRow('Steps:')
        self.guidance_scale = MetadataRow('CFG Scale:')
        self.size = MetadataRow('Size:')
        self.condition = MetadataRow('Condition:')
        self.control_net = MetadataRow('Control Net:')
        self.source_path = MetadataRow('Source Path:')
        self.img_strength = MetadataRow('Image Strength:')
        self.upscale = MetadataRow('Upscaling:')
        self.face = MetadataRow('Face Restoration:')

        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.path.frame)
        vlayout.addWidget(self.type.frame)
        vlayout.addWidget(self.scheduler.frame)
        vlayout.addWidget(self.model.frame)
        vlayout.addWidget(self.prompt.frame)
        vlayout.addWidget(self.negative_prompt.frame)
        vlayout.addWidget(self.seed.frame)
        vlayout.addWidget(self.num_inference_steps.frame)
        vlayout.addWidget(self.guidance_scale.frame)
        vlayout.addWidget(self.size.frame)
        vlayout.addWidget(self.condition.frame)
        vlayout.addWidget(self.control_net.frame)
        vlayout.addWidget(self.source_path.frame)
        vlayout.addWidget(self.img_strength.frame)
        vlayout.addWidget(self.upscale.frame)
        vlayout.addWidget(self.face.frame)
        vlayout.addStretch()

    def update(self, metadata):
        self.path.value.setText(metadata.path)
        self.type.value.setText(metadata.type)
        self.scheduler.value.setText(metadata.scheduler)
        self.model.value.setText(metadata.model)
        self.prompt.value.setText(metadata.prompt)
        self.negative_prompt.value.setText(metadata.negative_prompt)
        self.seed.value.setText(str(metadata.seed))
        self.num_inference_steps.value.setText(str(metadata.num_inference_steps))
        self.guidance_scale.value.setText(str(metadata.guidance_scale))
        self.size.value.setText('{:d}x{:d}'.format(metadata.width, metadata.height))
        self.condition.value.setText(metadata.condition)
        if metadata.type == 'img2img':
            condition = configuration.conditions.get(metadata.condition, None)
            if isinstance(condition, ControlNetCondition):
                self.img_strength.frame.setVisible(False)
                self.control_net.frame.setVisible(True)
                self.control_net.value.setText('Preprocess={:s}, Scale={:g}, {:s}'.format(
                    str(metadata.control_net_preprocess),
                    metadata.control_net_scale,
                    metadata.control_net_model
                ))
            else:
                self.control_net.frame.setVisible(False)
                self.img_strength.frame.setVisible(True)
                self.img_strength.value.setText(str(metadata.img_strength))
            self.source_path.frame.setVisible(True)
            self.source_path.value.setText(metadata.source_path)
        else:
            self.control_net.frame.setVisible(False)
            self.source_path.frame.setVisible(False)
            self.img_strength.frame.setVisible(False)
        if metadata.upscale_enabled:
            self.upscale.frame.setVisible(True)
            self.upscale.value.setText('{:d}x, Denoising={:g}, Blend={:g}'.format(
                metadata.upscale_factor,
                metadata.upscale_denoising_strength,
                metadata.upscale_blend_strength
            ))
        else:
            self.upscale.frame.setVisible(False)
        if metadata.face_enabled:
            self.face.frame.setVisible(True)
            self.face.value.setText('Blend={:g}'.format(
                metadata.face_blend_strength
            ))
        else:
            self.face.frame.setVisible(False)

class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(300)
        self.padding = 5
        self.minimum_image_size = 100
        self.both_images_visible = False
        self.show_preview = True

        self.left_image_path_ = ''
        self.right_image_path_ = ''

        self.left_image = None
        self.right_image = None
        self.preview_image = None

        self.left_label = QLabel(self)
        self.right_label = QLabel(self)
        self.left_controls_frame = QFrame(self)
        self.right_controls_frame = QFrame(self)
        self.metadata_frame = ImageMetadataFrame(self)
        self.metadata_frame.setVisible(False)

        self.locate_source_button = actions.locate_source.tool_button()
        self.send_to_img2img_button = actions.send_to_img2img.tool_button()
        self.use_prompt_button = actions.use_prompt.tool_button()
        self.use_seed_button = actions.use_seed.tool_button()
        self.use_initial_image_button = actions.use_initial_image.tool_button()
        self.use_all_button = actions.use_all.tool_button()
        self.toggle_metadata_button = actions.toggle_metadata.tool_button()
        self.toggle_metadata_button.toggled.connect(self.on_metadata_button_changed)
        self.toggle_preview_button = actions.toggle_preview.tool_button()
        self.toggle_preview_button.setChecked(True)
        self.toggle_preview_button.toggled.connect(self.on_preview_button_changed)
        self.delete_button = actions.delete_image.tool_button()

        left_controls_layout = QHBoxLayout(self.left_controls_frame)
        left_controls_layout.setContentsMargins(0, 0, 0, 0)
        left_controls_layout.setSpacing(0)
        left_controls_layout.addStretch()
        left_controls_layout.addWidget(self.locate_source_button)
        left_controls_layout.addStretch()

        right_controls_layout = QHBoxLayout(self.right_controls_frame)
        right_controls_layout.setContentsMargins(0, 0, 0, 0)
        right_controls_layout.setSpacing(0)
        right_controls_layout.addStretch()
        right_controls_layout.addWidget(self.send_to_img2img_button)
        right_controls_layout.addSpacing(8)
        right_controls_layout.addWidget(self.use_prompt_button)
        right_controls_layout.addWidget(self.use_seed_button)
        right_controls_layout.addWidget(self.use_initial_image_button)
        right_controls_layout.addWidget(self.use_all_button)
        right_controls_layout.addSpacing(8)
        right_controls_layout.addWidget(self.toggle_metadata_button)
        right_controls_layout.addWidget(self.toggle_preview_button)
        right_controls_layout.addSpacing(8)
        right_controls_layout.addWidget(self.delete_button)
        right_controls_layout.addStretch()

        background_color = QApplication.instance().palette().color(QPalette.Base)
        self.setStyleSheet(f'ImageViewer {{ background-color: {background_color.name()}; }}')
        self.setAttribute(Qt.WA_StyledBackground, True)

    def resizeEvent(self, event):
        self.update_images()

    def update_images(self):
        widget_width = self.width()
        widget_height = self.height()
        controls_height = 24

        use_preview_image = self.preview_image is not None and self.show_preview
        right_scale_factor = 1 if use_preview_image else self.metadata.upscale_factor if self.right_image else 1
        left_scale_factor = self.left_metadata.upscale_factor if self.left_image else 1
        right_image = self.preview_image if use_preview_image else self.right_image

        right_image_width = right_image.width() / right_scale_factor if right_image is not None else 1
        right_image_height = right_image.height() / right_scale_factor if right_image is not None else 1
        left_image_width = self.left_image.width() / left_scale_factor if self.left_image is not None else right_image_width
        left_image_height = self.left_image.height() / left_scale_factor if self.left_image is not None else right_image_height

        left_min_size = left_image_height // 2
        right_min_size = right_image_height // 2

        if self.both_images_visible:
            available_height = widget_height - controls_height - 4 * self.padding
            available_width = widget_width - 3 * self.padding

            right_height = min(available_height, right_image_height)
            right_width = int(right_image_width * (right_height / right_image_height))

            remaining_width = available_width - right_width
            left_width = min(remaining_width, left_image_width)
            left_height = int(left_image_height * (left_width / left_image_width))

            if left_height > available_height:
                left_height = available_height
                left_width = int(left_image_width * (left_height / left_image_height))

            if left_height < left_min_size:
                left_height = left_min_size
                left_width = int(left_image_width * (left_height / left_image_height))
                right_width = available_width - left_width
                right_height = int(right_image_height * (right_width / right_image_width))
                if right_height > available_height:
                    right_height = available_height
                    right_width = int(right_image_width * (right_height / right_image_height))
                if right_height < right_min_size:
                    right_height = right_min_size
                    right_width = int(right_image_width * (right_height / right_image_height))

            if self.left_image is not None:
                left_pixmap = QPixmap.fromImage(self.left_image).scaled(left_width, left_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.left_label.setPixmap(left_pixmap)
                self.left_label.setStyleSheet('')  
            else:
                self.left_label.setText('Choose a Source Image')
                self.left_label.setStyleSheet('border: 2px solid white;')
                self.left_label.setWordWrap(True)
                self.left_label.setAlignment(Qt.AlignCenter)

            if right_image is not None:
                right_pixmap = QPixmap.fromImage(right_image).scaled(right_width, right_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.right_label.setPixmap(right_pixmap)

            left_x = (widget_width - left_width - right_width - self.padding) // 2
            left_y = controls_height + 2 * self.padding + (available_height - left_height) // 2
            right_x = left_x + self.padding + left_width
            right_y = controls_height + 2 * self.padding + (available_height - right_height) // 2

            self.left_controls_frame.setVisible(True)
            self.left_label.setVisible(True)

            self.left_controls_frame.setGeometry(left_x, self.padding, left_width, controls_height)
            self.left_label.setGeometry(left_x, left_y, left_width, left_height)
            self.right_controls_frame.setGeometry(right_x, self.padding, right_width, controls_height)
            self.right_label.setGeometry(right_x, right_y, right_width, right_height)
            self.metadata_frame.setGeometry(right_x, right_y, right_width, right_height)
        else:
            available_height = widget_height - controls_height - 4 * self.padding
            available_width = widget_width - 2 * self.padding

            right_height = min(available_height, right_image_height)
            right_width = int(right_image_width * (right_height / right_image_height))

            if right_image is not None:
                right_pixmap = QPixmap.fromImage(right_image).scaled(right_width, right_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.right_label.setPixmap(right_pixmap)

            right_x = (widget_width - right_width) // 2
            right_y = controls_height + 2 * self.padding + (available_height - right_height) // 2

            self.left_controls_frame.setVisible(False)
            self.left_label.setVisible(False)

            self.right_controls_frame.setGeometry(right_x, self.padding, right_width, controls_height)
            self.right_label.setGeometry(right_x, right_y, right_width, right_height)
            self.metadata_frame.setGeometry(right_x, right_y, right_width, right_height)

    def set_both_images_visible(self, both_images_visible):
        self.both_images_visible = both_images_visible
        self.update_images()

    def left_image_path(self):
        return self.left_image_path_
    
    def right_image_path(self):
        return self.right_image_path_

    def clear_left_image(self):
        self.left_image_path_ = ''
        self.left_image = None
        self.update_images()

    def set_left_image(self, path):
        full_path = os.path.join(configuration.IMAGES_PATH, path)
        try:
            with Image.open(full_path) as image:
                self.left_metadata = ImageMetadata()
                self.left_metadata.path = path
                self.left_metadata.load_from_image_info(image.info)

                self.left_image_path_ = path
                self.left_image = utils.pil_to_qimage(image)
        except (IOError, OSError):
            self.left_image_path_ = ''
            self.left_image = None
        self.update_images()

    def set_right_image(self, path):
        full_path = os.path.join(configuration.IMAGES_PATH, path)
        try:
            with Image.open(full_path) as image:
                self.metadata = ImageMetadata()
                self.metadata.path = path
                self.metadata.load_from_image_info(image.info)

                self.right_image_path_ = path
                self.right_image = utils.pil_to_qimage(image)
        except (IOError, OSError):
            self.left_image_path_ = ''
            self.left_image = None

        self.metadata_frame.update(self.metadata)
        self.update_images()

    def set_preview_image(self, preview_image):
        if preview_image is not None:
            self.preview_image = utils.pil_to_qimage(preview_image)
        else:
            self.preview_image = None
        self.update_images()

    def on_metadata_button_changed(self, state):
        self.metadata_frame.setVisible(state)

    def on_preview_button_changed(self, state):
        self.show_preview = state
        self.update_images()
