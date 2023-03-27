import json
import os
import random
import re
import sys
import warnings

import numpy as np
import torch
from compel import Compel
from diffusers import (EulerAncestralDiscreteScheduler,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPipeline)
from PIL import Image, PngImagePlugin
from PySide6.QtCore import QSettings, QSize, Qt, QThread, Signal
from PySide6.QtGui import (QAction, QFontMetrics, QIcon, QImage, QPalette,
                           QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup,
                               QCheckBox, QDoubleSpinBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QSpinBox, QSplitter, QStyle, QStyleOptionSlider,
                               QToolBar, QToolButton, QVBoxLayout, QWidget)

if sys.platform == "darwin":
    from AppKit import NSApplication

warnings.filterwarnings('ignore')
from gfpgan import GFPGANer

warnings.resetwarnings()

# -------------------------------------------------------------------------------------------------
APP_NAME = 'SimpleDiffusion'
APP_VERSION = 0.1
IMAGES_PATH = 'images/outputs'
REPO_ID = 'stabilityai/stable-diffusion-2-1-base'

def set_default_setting(settings, key, value):
    if not settings.contains(key):
        settings.setValue(key, value)

def bool_setting(settings, key):
    value = settings.value(key)
    if value is not None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
    return False

def to_qimage(pil_image):
    data = pil_image.convert('RGBA').tobytes('raw', 'RGBA')
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    return qimage

class ImageMetadata:
    def __init__(self):
        self.mode = 'txt2img'
        self.path = ''
        self.prompt = ''
        self.negative_prompt = ''
        self.seed = 1
        self.num_inference_steps = 30
        self.guidance_scale = 7.5
        self.width = 512
        self.height = 512
        self.source_path = ''
        self.img_strength = 0.0
        self.gfpgan_enabled = False
        self.gfpgan_strength  = 0.0

    def load_from_settings(self, settings):
        self.mode = settings.value('mode')
        self.prompt = settings.value('prompt')
        self.negative_prompt = settings.value('negative_prompt')
        self.seed = int(settings.value('seed'))
        self.num_inference_steps = int(settings.value('num_inference_steps'))
        self.guidance_scale = float(settings.value('guidance_scale'))
        self.width = int(settings.value('width'))
        self.height = int(settings.value('height'))
        self.source_path = ''
        self.img_strength = 0.0
        if self.mode == 'img2img':
            self.source_path = settings.value('source_path')
            self.img_strength = float(settings.value('img_strength'))
        self.gfpgan_enabled = bool_setting(settings, 'gfpgan_enabled')
        self.gfpgan_strength = 0.0
        if self.gfpgan_enabled:
            self.gfpgan_strength = float(settings.value('gfpgan_strength'))
    
    def load_from_png_info(self, image_info):
        if 'sd-metadata' in image_info:
            data = json.loads(image_info['sd-metadata'])
            if 'image' in data:
                image_data = data['image']
                self.mode = image_data.get('type', 'txt2img')
                self.prompt = image_data.get('prompt', '')
                self.negative_prompt = image_data.get('negative_prompt', '')
                self.seed = int(image_data.get('seed', 5))
                self.steps = int(image_data.get('steps', 30))
                self.guidance_scale = float(image_data.get('cfg_scale', 7.5))
                self.width = int(image_data.get('width', 512))
                self.height = int(image_data.get('height', 512))
                self.source_path = ''
                self.img_strength = 0.0
                if self.mode == 'img2img':
                    self.source_path = image_data.get('source_path', '')
                    self.img_strength = float(image_data.get('img_strength', 0.5))
                self.gfpgan_enabled = 'gfpgan_strength' in image_data
                self.gfpgan_strength = 0.0
                if self.gfpgan_enabled:
                    self.gfpgan_strength = float(image_data.get('gfpgan_strength'))

    def save_to_png_info(self, png_info):
        sd_metadata = {
            'model': 'stable diffusion',
            'model_weights': REPO_ID,
            'model_hash': '',    # TODO
            'app_id': APP_NAME,
            'APP_VERSION': APP_VERSION,
            'image': {
                'prompt': self.prompt,
                'negative_prompt': self.negative_prompt,
                'steps': str(self.num_inference_steps),
                'cfg_scale': str(self.guidance_scale),
                'height': str(self.height),
                'width': str(self.width),
                'seed': str(self.seed),
                'type': self.mode,
                'sampler': 'k_euler_a',
            }
        }
        if self.mode == 'img2img':
            sd_metadata['image']['source_path'] = self.source_path
            sd_metadata['image']['img_strength'] = self.img_strength
        if self.gfpgan_enabled:
            sd_metadata['image']['gfpgan_strength'] = self.gfpgan_strength

        png_info.add_text('Dream',
            '"{:s} [{:s}]" -s {:d} -S {:d} -W {:d} -H {:d} -C {:f} -A k_euler_a'.format(
                self.prompt, self.negative_prompt, self.num_inference_steps, self.seed, self.width, self.height, self.guidance_scale
            ))
        png_info.add_text('sd-metadata', json.dumps(sd_metadata))


class PlaceholderTextEdit(QPlainTextEdit):
    def __init__(self, desired_lines, placeholder_text, parent=None):
        super().__init__(parent)

        font = self.font()
        font.setPointSize(14)
        self.setFont(font)

        font_metrics = QFontMetrics(font)
        line_height = font_metrics.lineSpacing()
        margins = self.contentsMargins()
        frame_width = self.frameWidth()
        document_margins = self.document().documentMargin()

        self.setFixedHeight(line_height * desired_lines + margins.top() + margins.bottom() + 2 * frame_width + 2 * document_margins)
        self.setPlaceholderText(placeholder_text)
        self.setTabChangesFocus(True)

class ThumbnailViewer(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.action_use_prompt = QAction('Use Prompt')
        self.action_use_seed = QAction('Use Seed')
        self.action_use_all = QAction('Use All')
        self.action_use_initial_image = QAction('Use Initial Image')
        self.action_send_to_img2img = QAction('Send to Image to Image')
        self.action_delete = QAction('Delete Image')

        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setSpacing(10)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.min_thumbnail_size = 100
        self.max_thumbnail_size = 250
        self.margin = 0

        self.menu = QMenu()
        self.menu.addAction(self.action_use_prompt)
        self.menu.addAction(self.action_use_seed)
        self.menu.addAction(self.action_use_all)
        self.menu.addAction(self.action_use_initial_image)
        self.menu.addAction(self.action_send_to_img2img)
        self.menu.addAction(self.action_delete)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_icon_size()

    def visualRect(self, index):
        rect = super().visualRect(index)
        rect.setWidth(self.iconSize().width())
        rect.setHeight(self.iconSize().height())
        return rect

    def add_thumbnail(self, path):
        full_path = os.path.join(IMAGES_PATH, path)
        image = Image.open(full_path)
        metadata = ImageMetadata()
        metadata.path = path
        metadata.load_from_png_info(image.info)
        pixmap = QPixmap.fromImage(to_qimage(image))
        scaled_pixmap = pixmap.scaled(self.max_thumbnail_size, self.max_thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon = QIcon(scaled_pixmap)
        item = QListWidgetItem()
        item.setIcon(icon)
        item.setData(Qt.UserRole, metadata)
        self.insertItem(0, item)

    def update_icon_size(self):
        style = QApplication.instance().style()
        scrollbar_width = style.pixelMetric(QStyle.PM_ScrollBarExtent, QStyleOptionSlider())
        available_width = self.width() - scrollbar_width
        num_columns = int((available_width) / (self.min_thumbnail_size))
        num_columns = max(1, num_columns)
        new_icon_size = int((available_width - num_columns) / num_columns)
        new_icon_size = max(self.min_thumbnail_size, min(new_icon_size, self.max_thumbnail_size))

        self.setIconSize(QSize(new_icon_size, new_icon_size))
        self.setGridSize(QSize(new_icon_size, new_icon_size))

    def show_context_menu(self, point):
        self.menu.exec(self.mapToGlobal(point))

class ImageViewer(QWidget):
    def __init__(self, both_images_visible, parent=None):
        super().__init__(parent)
        self.padding = 5
        self.minimum_image_size = 100
        self.both_images_visible = both_images_visible

        self.left_image_path_ = ''
        self.right_image_path_ = ''

        self.left_image = None
        self.right_image = None

        self.left_label = QLabel(self)
        self.right_label = QLabel(self)

        background_color = QApplication.instance().palette().color(QPalette.Base)
        self.setStyleSheet(f'background-color: {background_color.name()};')
        self.setAttribute(Qt.WA_StyledBackground, True)

    def resizeEvent(self, event):
        self.update_images()

    def update_images(self):
        window_width = self.width()
        window_height = self.height()

        left_image_width = self.left_image.width() if self.left_image is not None else 1
        left_image_height = self.left_image.height() if self.left_image is not None else 1
        right_image_width = self.right_image.width() if self.right_image is not None else 1
        right_image_height = self.right_image.height() if self.right_image is not None else 1

        if self.both_images_visible:
            available_height = window_height - 2 * self.padding
            available_width = window_width - 3 * self.padding

            right_height = min(available_height, right_image_height)
            right_width = int(right_image_width * (right_height / right_image_height))

            remaining_width = available_width - right_width
            left_width = min(remaining_width, left_image_width)
            left_height = int(left_image_height * (left_width / left_image_width))

            if left_height > available_height:
                left_height = available_height
                left_width = int(left_image_width * (left_height / left_image_height))

            if left_height < self.minimum_image_size:
                left_height = self.minimum_image_size
                left_width = int(left_image_width * (left_height / left_image_height))
                right_width = available_width - left_width
                right_height = int(right_image_height * (right_width / right_image_width))
                if right_height > available_height:
                    right_height = available_height
                    right_width = int(right_image_width * (right_height / right_image_height))
                if right_height < self.minimum_image_size:
                    right_height = self.minimum_image_size
                    right_width = int(right_image_width * (right_height / right_image_height))

            if self.left_image is not None:
                left_pixmap = QPixmap.fromImage(self.left_image).scaled(left_width, left_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.left_label.setPixmap(left_pixmap)

            if self.right_image is not None:
                right_pixmap = QPixmap.fromImage(self.right_image).scaled(right_width, right_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.right_label.setPixmap(right_pixmap)

            left_x = (window_width - left_width - right_width - self.padding) // 2
            right_x = left_x + self.padding + left_width

            self.left_label.setVisible(True)
            self.left_label.setGeometry(left_x, (window_height - left_height) // 2, left_width, left_height)
            self.right_label.setGeometry(right_x, (window_height - right_height) // 2, right_width, right_height)
        else:
            available_height = window_height - 2 * self.padding
            available_width = window_width - 2 * self.padding

            right_height = min(available_height, right_image_height)
            right_width = int(right_image_width * (right_height / right_image_height))

            if self.right_image is not None:
                right_pixmap = QPixmap.fromImage(self.right_image).scaled(right_width, right_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.right_label.setPixmap(right_pixmap)

            right_x = (window_width - right_width) // 2

            self.left_label.setVisible(False)
            self.right_label.setGeometry(right_x, (window_height - right_height) // 2, right_width, right_height)

    def set_both_images_visible(self, both_images_visible):
        self.both_images_visible = both_images_visible
        self.update_images()

    def left_image_path(self):
        return self.left_image_path_
    
    def right_image_path(self):
        return self.right_image_path_

    def set_left_image(self, path):
        self.left_image_path_ = path
        self.left_image = QImage(os.path.join(IMAGES_PATH, path))
        self.update_images()

    def set_right_image(self, path):
        self.right_image_path_ = path
        self.right_image = QImage(os.path.join(IMAGES_PATH, path))
        self.update_images()

class FloatSliderSpinBox(QWidget):
    def __init__(self, name, initial_value, checkable=False, parent=None):
        super().__init__(parent)

        if checkable:
            self.check_box = QCheckBox(name)
            self.check_box.setChecked(True)
            self.check_box.stateChanged.connect(self.on_check_box_changed)
        else:
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(initial_value * 100)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setAlignment(Qt.AlignCenter)
        self.spin_box.setFixedWidth(80)
        self.spin_box.setRange(0.01, 1.0)
        self.spin_box.setSingleStep(0.01)
        self.spin_box.setDecimals(2)
        self.spin_box.setValue(initial_value)
        self.spin_box.valueChanged.connect(self.on_spin_box_changed)

        hlayout = QHBoxLayout(frame)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spin_box)

        vlayout = QVBoxLayout(self)
        vlayout.setContentsMargins(0, 0, 0, 0) 
        vlayout.setSpacing(0)
        if checkable:
            check_box_layout = QHBoxLayout()
            check_box_layout.setAlignment(Qt.AlignCenter)
            check_box_layout.addWidget(self.check_box)
            vlayout.addLayout(check_box_layout)
        else:
            vlayout.addWidget(label)
        vlayout.addWidget(frame)

    def on_check_box_changed(self, state):
        self.slider.setEnabled(state)
        self.spin_box.setEnabled(state)

    def on_slider_changed(self, value):
        decimal_value = value / 100
        self.spin_box.setValue(decimal_value)

    def on_spin_box_changed(self, value):
        slider_value = int(value * 100)
        self.slider.setValue(slider_value)

class InitThread(QThread):
    task_complete = Signal(dict, GFPGANer)

    def __init__(self, parent=None):
        super().__init__(parent)
 
    def run(self):
        warnings.filterwarnings('ignore')

        dtype = torch.float32
        device = 'mps'
        base_pipe = StableDiffusionPipeline.from_pretrained(REPO_ID, safety_checker=None, torch_dtype=dtype, requires_safety_checker=False)

        pipes = {}
        pipes['txt2img'] = base_pipe
        pipes['img2img'] = StableDiffusionImg2ImgPipeline(**base_pipe.components, requires_safety_checker=False)

        for pipe in pipes.values():
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.enable_attention_slicing()
            pipe = pipe.to(device)

        gfpgan = GFPGANer(
            model_path='data/GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )

        warnings.resetwarnings()

        self.task_complete.emit(pipes, gfpgan)

class GenerateThread(QThread):
    update_progress = Signal(int)
    image_complete = Signal(str)
    task_complete = Signal(int)

    def __init__(self, next_image_id, mode, pipes, gfpgan, settings, parent=None):
        super().__init__(parent)

        self.next_image_id = next_image_id
        self.mode = mode
        self.pipes = pipes
        self.gfpgan = gfpgan
        self.metadata = ImageMetadata()
        self.metadata.load_from_settings(settings)
        self.num_images_per_prompt = int(settings.value('num_images_per_prompt', 1))

    def run(self):
        # mode
        pipe = self.pipes[self.mode]
        if self.mode == 'img2img':
            image_path = os.path.join(IMAGES_PATH, self.metadata.source_path)
            f = open(image_path, 'rb')
            source_image = Image.open(f).convert('RGB')
            source_image = source_image.resize((self.metadata.width, self.metadata.height))
            f.close()

        # generator
        generator = torch.Generator().manual_seed(self.metadata.seed)

        # prompt weighting
        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        prompt_embeds = compel_proc(self.metadata.prompt)
        negative_prompt_embeds = compel_proc(self.metadata.negative_prompt)

        # generate
        if self.mode == 'txt2img':
            images = pipe(
                width=self.metadata.width,
                height=self.metadata.height,
                num_inference_steps=self.metadata.num_inference_steps,
                guidance_scale=self.metadata.guidance_scale,
                num_images_per_prompt=self.num_images_per_prompt,
                generator=generator,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                callback=self.generate_callback,
            ).images
        elif self.mode == 'img2img':
            images = pipe(
                image=source_image,
                strength=self.metadata.img_strength,
                num_inference_steps=self.metadata.num_inference_steps,
                guidance_scale=self.metadata.guidance_scale,
                num_images_per_prompt=self.num_images_per_prompt,
                generator=generator,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                callback=self.generate_callback,
            ).images

        steps = self.compute_total_steps()
        step = steps - self.num_images_per_prompt
        for image in images:
            # GFPGAN
            if self.metadata.gfpgan_strength > 0.0:
                bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

                _, _, restored_img = self.gfpgan.enhance(
                    bgr_image_array,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )

                image2 = Image.fromarray(restored_img[..., ::-1])

                if self.metadata.gfpgan_strength < 1.0:
                    if image2.size != image.size:
                        image = image.resize(image2.size)
                    image = Image.blend(image, image2, self.metadata.gfpgan_strength)
                else:
                    image = image2

            progress_amount = (step+1) * 100 / steps
            step = step + 1
            self.update_progress.emit(progress_amount)

            # Output
            output_file = '{:05d}.png'.format(self.next_image_id)
            self.next_image_id = self.next_image_id + 1
            output_path = os.path.join(IMAGES_PATH, output_file)

            png_info = PngImagePlugin.PngInfo()
            self.metadata.save_to_png_info(png_info)
            image.save(output_path, pnginfo=png_info)

            self.image_complete.emit(output_file)

        self.task_complete.emit(self.next_image_id)

    def generate_callback(self, step: int, timestep: int, latents: torch.FloatTensor):
        steps = self.compute_total_steps()
        progress_amount = (step+1) * 100 / steps
        self.update_progress.emit(progress_amount)

    def compute_total_steps(self):
        if self.mode == 'img2img':
            steps = int(self.metadata.num_inference_steps * self.metadata.img_strength)
        else:
            steps = self.metadata.num_inference_steps
        steps = steps + self.num_images_per_prompt
        return steps

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Settings
        self.settings = QSettings('settings.ini', QSettings.IniFormat)
        set_default_setting(self.settings, 'mode', 'txt2img')
        set_default_setting(self.settings, 'prompt', '')
        set_default_setting(self.settings, 'negative_prompt', '')
        set_default_setting(self.settings, 'manual_seed', False)
        set_default_setting(self.settings, 'seed', 1)
        set_default_setting(self.settings, 'num_images_per_prompt', 1)
        set_default_setting(self.settings, 'num_inference_steps', 30)
        set_default_setting(self.settings, 'guidance_scale', 7.5)
        set_default_setting(self.settings, 'width', 512)
        set_default_setting(self.settings, 'height', 512)
        set_default_setting(self.settings, 'source_path', '')
        set_default_setting(self.settings, 'img_strength', 0.5)
        set_default_setting(self.settings, 'gfpgan_enabled', False)
        set_default_setting(self.settings, 'gfpgan_strength', 0.8)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Modes
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)

        txt2img_button = QToolButton()
        txt2img_button.setIcon(QIcon('data/txt2img_icon.png'))
        txt2img_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        txt2img_button.setStyleSheet('QToolButton:checked { background-color: darkblue; }')
        txt2img_button.setCheckable(True)
        txt2img_button.setAutoExclusive(True)
        txt2img_button.setToolTip('Text To Image')

        img2img_button = QToolButton()
        img2img_button.setIcon(QIcon('data/img2img_icon.png'))
        img2img_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        img2img_button.setStyleSheet('QToolButton:checked { background-color: darkblue; }')
        img2img_button.setCheckable(True)
        img2img_button.setAutoExclusive(True)
        img2img_button.setToolTip('Image To Image')

        toolbar.addWidget(txt2img_button)
        toolbar.addWidget(img2img_button)

        self.button_group = QButtonGroup()
        self.button_group.addButton(txt2img_button, 0)
        self.button_group.addButton(img2img_button, 1)
        self.button_group.idToggled.connect(self.on_mode_changed)

        # Configuration controls
        config_frame = QFrame()
        config_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        config_frame.setContentsMargins(0, 0, 0, 0)

        self.prompt_edit = PlaceholderTextEdit(8, 'Prompt')
        self.prompt_edit.setPlainText(self.settings.value('prompt'))
        self.negative_prompt_edit = PlaceholderTextEdit(3, 'Negative Prompt')
        self.negative_prompt_edit.setPlainText(self.settings.value('negative_prompt'))

        self.generate_button = QPushButton('Generate')
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.on_generate_clicked)

        controls_frame = QFrame()
        controls_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_grid = QGridLayout()
        controls_grid.setContentsMargins(0, 0, 0, 0)
        controls_grid.setVerticalSpacing(2)
        controls_grid.setRowMinimumHeight(2, 10)
        controls_frame.setLayout(controls_grid)

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
        self.guidance_scale_spin_box.setMinimum(0.5)
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

        self.manual_seed_check_box = QCheckBox('Manual Seed')

        self.seed_frame = QFrame()
        self.seed_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.seed_lineedit = QLineEdit()
        self.seed_lineedit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.seed_lineedit.setText(str(self.settings.value('seed')))
        seed_random_button = QPushButton('New')
        seed_random_button.clicked.connect(self.on_seed_random_clicked)

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

        manual_seed = bool_setting(self.settings, 'manual_seed')
        self.seed_frame.setEnabled(manual_seed)
        self.manual_seed_check_box.setChecked(manual_seed)
        self.manual_seed_check_box.stateChanged.connect(self.on_manual_seed_check_box_changed)

        self.img_strength = FloatSliderSpinBox('Image Strength', float(self.settings.value('img_strength')))
        self.img_strength.setVisible(False)

        self.gfpgan_strength = FloatSliderSpinBox('Face Restoration', float(self.settings.value('gfpgan_strength')), checkable=True)
        self.gfpgan_strength.check_box.setChecked(bool_setting(self.settings, 'gfpgan_enabled'))

        config_layout = QVBoxLayout(config_frame)
        config_layout.setContentsMargins(0, 0, 0, 0) 
        config_layout.addWidget(self.prompt_edit)
        config_layout.addWidget(self.negative_prompt_edit)
        config_layout.addWidget(self.generate_button)
        config_layout.addWidget(controls_frame)
        config_layout.addLayout(seed_vlayout)
        config_layout.addWidget(self.img_strength)
        config_layout.addWidget(self.gfpgan_strength)
        config_layout.addStretch()

        # Image viewer
        image_frame = QFrame()
        image_frame.setContentsMargins(0, 0, 0, 0)
        self.image_viewer = ImageViewer(False)
        self.image_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.image_viewer)

        #  Thumbnail viewer
        thumbnail_frame = QFrame()
        thumbnail_frame.setContentsMargins(0, 0, 0, 0)

        self.thumbnail_viewer = ThumbnailViewer()
        self.thumbnail_viewer.itemSelectionChanged.connect(self.on_thumbnail_selection_change)
        self.thumbnail_viewer.action_use_prompt.triggered.connect(self.on_use_prompt)
        self.thumbnail_viewer.action_use_seed.triggered.connect(self.on_use_seed)
        self.thumbnail_viewer.action_use_all.triggered.connect(self.on_use_all)
        self.thumbnail_viewer.action_use_initial_image.triggered.connect(self.on_use_initial_image)
        self.thumbnail_viewer.action_send_to_img2img.triggered.connect(self.on_send_to_img2img)
        self.thumbnail_viewer.action_delete.triggered.connect(self.on_delete)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area.setWidget(self.thumbnail_viewer)

        thumbnail_layout = QVBoxLayout(thumbnail_frame)
        thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        thumbnail_layout.addWidget(scroll_area)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(image_frame)
        splitter.addWidget(thumbnail_frame)
        splitter.setStretchFactor(0, 1)  # left widget
        splitter.setStretchFactor(1, 0)  # right widget

        palette = QApplication.instance().palette()
        background_color = palette.color(QPalette.Window)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet('QProgressBar::chunk {{ background-color: blue; }}')
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)

        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(8, 2, 8, 8)
        hlayout.setSpacing(8)
        hlayout.addWidget(config_frame)
        hlayout.addWidget(splitter)

        vlayout = QVBoxLayout(central_widget)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        vlayout.addWidget(self.progress_bar)
        vlayout.addLayout(hlayout)

        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, 1200, 600)

        # Gather images
        if self.settings.value('source_path') != '':
            self.image_viewer.set_left_image(self.settings.value('source_path'))

        image_files = sorted([file for file in os.listdir(IMAGES_PATH) if file.lower().endswith(('.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

        if len(image_files) > 0:
            self.image_viewer.set_right_image(image_files[-1])

        self.next_image_id = 0
        for image_file in image_files:
            match = re.match(r'(\d+)\.png', image_file)
            if match:
                self.next_image_id = max(self.next_image_id, int(match.group(1)))
            self.thumbnail_viewer.add_thumbnail(image_file)
        self.thumbnail_viewer.setCurrentRow(0)
        self.next_image_id = self.next_image_id + 1

        # Apply settings that impact other controls
        self.set_mode(self.settings.value('mode'))

        # Initialize pipelines
        self.init_thread = InitThread(self)
        self.init_thread.task_complete.connect(self.init_complete)
        self.init_thread.start()

    def init_complete(self, pipes, gfpgan):
        self.pipes = pipes
        self.gfpgan = gfpgan
        self.generate_button.setEnabled(True)

    def set_mode(self, mode):
        self.mode = mode
        if self.mode == 'txt2img':
            self.button_group.button(0).setChecked(True)
        elif self.mode == 'img2img':
            self.button_group.button(1).setChecked(True)

    def on_mode_changed(self, button_id, checked):
        if not checked:
            return
        if button_id == 0:
            self.mode = 'txt2img'
            self.img_strength.setVisible(False)
            self.image_viewer.set_both_images_visible(False)
        elif button_id == 1:
            self.mode = 'img2img'
            self.img_strength.setVisible(True)
            self.image_viewer.set_both_images_visible(True)

    def on_generate_clicked(self):
        if not self.manual_seed_check_box.isChecked():
            self.randomize_seed()

        self.settings.setValue('mode', self.mode)
        self.settings.setValue('prompt', self.prompt_edit.toPlainText())
        self.settings.setValue('negative_prompt', self.negative_prompt_edit.toPlainText())
        self.settings.setValue('manual_seed', self.manual_seed_check_box.isChecked())
        self.settings.setValue('seed', self.seed_lineedit.text())
        self.settings.setValue('num_images_per_prompt', self.num_images_spin_box.value())
        self.settings.setValue('num_inference_steps', self.num_steps_spin_box.value())
        self.settings.setValue('guidance_scale', self.guidance_scale_spin_box.value())
        self.settings.setValue('width', self.width_spin_box.value())
        self.settings.setValue('height', self.height_spin_box.value())
        self.settings.setValue('source_path', self.image_viewer.left_image_path())
        self.settings.setValue('img_strength', self.img_strength.spin_box.value())
        self.settings.setValue('gfpgan_enabled', self.gfpgan_strength.check_box.isChecked())
        self.settings.setValue('gfpgan_strength', self.gfpgan_strength.spin_box.value())

        self.generate_button.setEnabled(False)

        self.generate_thread = GenerateThread(
            next_image_id = self.next_image_id,
            mode = self.mode,
            pipes = self.pipes,
            gfpgan = self.gfpgan,
            settings = self.settings,
            parent = self
        )
        self.generate_thread.update_progress.connect(self.update_progress)
        self.generate_thread.image_complete.connect(self.image_complete)
        self.generate_thread.task_complete.connect(self.generate_complete)
        self.generate_thread.start()
    
    def update_progress(self, progress_amount):
        self.progress_bar.setValue(progress_amount)
        if sys.platform == "darwin":
            sharedApplication = NSApplication.sharedApplication()
            dockTile = sharedApplication.dockTile()
            if progress_amount > 0:
                dockTile.setBadgeLabel_('{:d}%'.format(progress_amount))
            else:
                dockTile.setBadgeLabel_(None)

    def image_complete(self, output_file):
        self.thumbnail_viewer.add_thumbnail(output_file)
        self.thumbnail_viewer.setCurrentRow(0)
        self.image_viewer.set_right_image(output_file)

    def generate_complete(self, next_image_id):
        self.next_image_id = next_image_id
        self.generate_button.setEnabled(True)
        self.update_progress(0)

    def randomize_seed(self):
        seed = random.randint(0, 0x7fff_ffff_ffff_ffff)
        self.seed_lineedit.setText(str(seed))

    def on_manual_seed_check_box_changed(self, state):
        self.seed_frame.setEnabled(state)

    def on_seed_random_clicked(self):
        self.randomize_seed()

    def on_thumbnail_selection_change(self):
        selected_items = self.thumbnail_viewer.selectedItems()
        for item in selected_items:
            image_metadata = item.data(Qt.UserRole)
            self.image_viewer.set_right_image(image_metadata.path)

    def get_current_metadata(self):
        item = self.thumbnail_viewer.currentItem()
        if item is not None:
            return item.data(Qt.UserRole)
        return None

    def on_use_prompt(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.prompt_edit.setPlainText(image_metadata.prompt)
            self.negative_prompt_edit.setPlainText(image_metadata.negative_prompt)

    def on_use_seed(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.manual_seed_check_box.setChecked(True)
            self.seed_lineedit.setText(str(image_metadata.seed))

    def on_use_all(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.prompt_edit.setPlainText(image_metadata.prompt)
            self.negative_prompt_edit.setPlainText(image_metadata.negative_prompt)
            self.manual_seed_check_box.setChecked(True)
            self.seed_lineedit.setText(str(image_metadata.seed))
            self.num_steps_spin_box.setValue(image_metadata.num_inference_steps)
            self.guidance_scale_spin_box.setValue(image_metadata.guidance_scale)
            self.width_spin_box.setValue(image_metadata.width)
            self.height_spin_box.setValue(image_metadata.height)
            if image_metadata.mode == 'img2img':
                self.image_viewer.set_left_image(image_metadata.source_path)
                self.img_strength.spin_box.setValue(image_metadata.img_strength)
            if image_metadata.gfpgan_enabled:
                self.gfpgan_strength.check_box.setChecked(True)
                self.gfpgan_strength.spin_box.setValue(image_metadata.gfpgan_strength)
            else:
                self.gfpgan_strength.check_box.setChecked(False)
            self.set_mode(image_metadata.mode)

    def on_use_initial_image(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.image_viewer.set_left_image(image_metadata.source_path)
            self.img_strength.spin_box.setValue(image_metadata.img_strength)
            self.set_mode('img2img')

    def on_send_to_img2img(self):
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            self.image_viewer.set_left_image(image_metadata.path)
            self.set_mode('img2img')
 
    def on_delete(self):
        item = self.thumbnail_viewer.currentItem()
        image_metadata = self.get_current_metadata()
        if image_metadata is not None:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Warning)
            message_box.setWindowTitle('Confirm Delete')
            message_box.setText('Are you sure you want to delete this image?')
            message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            message_box.setDefaultButton(QMessageBox.No)

            result = message_box.exec()
            if result == QMessageBox.Yes:
                image_path = os.path.join(IMAGES_PATH, image_metadata.path)
                os.remove(image_path)

                self.thumbnail_viewer.takeItem(self.thumbnail_viewer.row(item))

def main():
    os.makedirs(IMAGES_PATH, exist_ok=True)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('data/app_icon.png'))
    app.setApplicationName(APP_NAME)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
