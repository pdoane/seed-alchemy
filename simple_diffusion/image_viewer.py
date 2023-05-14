import json
import os

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from . import actions, configuration, utils
from .image_metadata import ImageMetadata


class MetadataRow:
    def __init__(self, label_text):
        self.label = QLabel(label_text)
        self.label.setStyleSheet('font-weight: bold; background-color: transparent')

        self.value = QLabel()
        self.value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.value.setStyleSheet('background-color: transparent')
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.value.setCursor(Qt.IBeamCursor)
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
        self.model = MetadataRow('Model:')
        self.scheduler = MetadataRow('Scheduler:')
        self.prompt = MetadataRow('Prompt:')
        self.negative_prompt = MetadataRow('Negative Prompt:')
        self.seed = MetadataRow('Seed:')
        self.num_inference_steps = MetadataRow('Steps:')
        self.guidance_scale = MetadataRow('CFG Scale:')
        self.size = MetadataRow('Size:')
        self.img2img = MetadataRow('Image to Image:')
        self.control_net = MetadataRow('Control Net:')
        self.upscale = MetadataRow('Upscaling:')
        self.face = MetadataRow('Face Restoration:')
        self.high_res = MetadataRow('High Resolution:')

        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.path.frame)
        vlayout.addWidget(self.scheduler.frame)
        vlayout.addWidget(self.model.frame)
        vlayout.addWidget(self.prompt.frame)
        vlayout.addWidget(self.negative_prompt.frame)
        vlayout.addWidget(self.seed.frame)
        vlayout.addWidget(self.num_inference_steps.frame)
        vlayout.addWidget(self.guidance_scale.frame)
        vlayout.addWidget(self.size.frame)
        vlayout.addWidget(self.img2img.frame)
        vlayout.addWidget(self.control_net.frame)
        vlayout.addWidget(self.upscale.frame)
        vlayout.addWidget(self.face.frame)
        vlayout.addWidget(self.high_res.frame)
        vlayout.addStretch()

    def update(self, metadata):
        self.path.value.setText(metadata.path)
        self.scheduler.value.setText(metadata.scheduler)
        self.model.value.setText(metadata.model)
        self.prompt.value.setText(metadata.prompt)
        self.negative_prompt.value.setText(metadata.negative_prompt)
        self.seed.value.setText(str(metadata.seed))
        self.num_inference_steps.value.setText(str(metadata.num_inference_steps))
        self.guidance_scale.value.setText(str(metadata.guidance_scale))
        self.size.value.setText('{:d}x{:d}'.format(metadata.width, metadata.height))

        if metadata.img2img_enabled:
            self.img2img.frame.setVisible(True)
            self.img2img.value.setText('Source={:s}, Blend={:.2f}'.format(
                metadata.img2img_source,
                metadata.img2img_strength,
            ))
        else:
            self.img2img.frame.setVisible(False)

        if metadata.control_net_enabled:
            self.control_net.frame.setVisible(True)
            self.control_net.value.setText('Range=[{:.2f},{:.2f}], {:s}'.format(
                metadata.control_net_guidance_start,
                metadata.control_net_guidance_end,
                json.dumps([control_net.to_dict() for control_net in metadata.control_nets])
            ))
        else:
            self.control_net.frame.setVisible(False)
        
        if metadata.upscale_enabled:
            self.upscale.frame.setVisible(True)
            self.upscale.value.setText('{:d}x, Denoising={:.2f}, Blend={:.2f}'.format(
                metadata.upscale_factor,
                metadata.upscale_denoising_strength,
                metadata.upscale_blend_strength
            ))
        else:
            self.upscale.frame.setVisible(False)

        if metadata.face_enabled:
            self.face.frame.setVisible(True)
            self.face.value.setText('Blend={:.2f}'.format(
                metadata.face_blend_strength
            ))
        else:
            self.face.frame.setVisible(False)
        
        if metadata.high_res_enabled:
            self.high_res.frame.setVisible(True)
            self.high_res.value.setText('Factor={:.2f}, Steps={:d}, Guidance={:.2f}, Noise={:.2f}'.format(
                metadata.high_res_factor,
                metadata.high_res_steps,
                metadata.high_res_guidance_scale,
                metadata.high_res_noise
            ))
        else:
            self.high_res.frame.setVisible(False)

class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(300)
        self.padding = 5
        self.minimum_image_size = 100
        self.show_preview = True

        self.image_path_ = ''

        self.image = None
        self.preview_image = None

        self.label = QLabel(self)
        self.controls_frame = QFrame()
        self.controls_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.controls_frame.setFrameStyle(QFrame.Panel)
        self.metadata_frame = ImageMetadataFrame(self)
        self.metadata_frame.setVisible(False)

        self.send_to_img2img_button = actions.send_to_img2img.tool_button()
        self.send_to_img2img_button.setPopupMode(QToolButton.InstantPopup)
        self.use_prompt_button = actions.use_prompt.tool_button()
        self.use_seed_button = actions.use_seed.tool_button()
        self.use_source_images_button = actions.use_source_images.tool_button()
        self.use_all_button = actions.use_all.tool_button()
        self.toggle_metadata_button = actions.toggle_metadata.tool_button()
        self.toggle_metadata_button.toggled.connect(self.on_metadata_button_changed)
        self.toggle_preview_button = actions.toggle_preview.tool_button()
        self.toggle_preview_button.setChecked(True)
        self.toggle_preview_button.toggled.connect(self.on_preview_button_changed)
        self.delete_button = actions.delete_image.tool_button()

        controls_layout = QHBoxLayout(self.controls_frame)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(0)
        controls_layout.addStretch()
        controls_layout.addWidget(self.send_to_img2img_button)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.use_prompt_button)
        controls_layout.addWidget(self.use_seed_button)
        controls_layout.addWidget(self.use_source_images_button)
        controls_layout.addWidget(self.use_all_button)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.toggle_metadata_button)
        controls_layout.addWidget(self.toggle_preview_button)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.delete_button)
        controls_layout.addStretch()

        widget_layout = QVBoxLayout(self)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        widget_layout.addWidget(self.controls_frame)
        widget_layout.addStretch()

    def resizeEvent(self, event):
        self.update_images()

    def update_images(self):
        widget_width = self.width()
        widget_height = self.height()
        controls_height = 32
        available_height = widget_height - controls_height - 4 * self.padding
        available_width = widget_width - 2 * self.padding

        use_preview_image = self.preview_image is not None and self.show_preview
        image = self.preview_image if use_preview_image else self.image
        image_width = image.width() if image else 1
        image_height = image.height() if image else 1

        width = image_width
        height = image_height

        if width > available_width:
            width = available_width
            height = int(image_height * (width / image_width))

        if height > available_height:
            height = available_height
            width = int(image_width * (height / image_height))

        if image is not None:
            pixmap = QPixmap.fromImage(image).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

        x = (widget_width - width) // 2
        y = controls_height + 2 * self.padding + (available_height - height) // 2

        self.label.setGeometry(x, y, width, height)
        self.metadata_frame.setGeometry(x, y, width, height)

    def set_image(self, path):
        full_path = os.path.join(configuration.IMAGES_PATH, path)
        try:
            with Image.open(full_path) as image:
                self.metadata = ImageMetadata()
                self.metadata.path = path
                self.metadata.load_from_image(image)

                self.image_path_ = path
                self.image = utils.pil_to_qimage(image)
        except (IOError, OSError):
            self.image_path_ = ''
            self.image = None

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
