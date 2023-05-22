import os

import qdarktheme
import torch
from PySide6.QtCore import QEvent, QSettings
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from . import configuration
from . import font_awesome as fa
from .main_window import MainWindow


class Application(QApplication):
    settings: QSettings = None
    collections: list[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dpi = QApplication.primaryScreen().logicalDotsPerInch()
        configuration.font_scale_factor = 96 / dpi

        # Directories
        os.makedirs(configuration.IMAGES_PATH, exist_ok=True)
        os.makedirs(configuration.MODELS_PATH, exist_ok=True)    

        # Settings
        self.settings = QSettings('settings.ini', QSettings.IniFormat)
        self.set_default_setting('local_models_path', '')
        self.set_default_setting('reduce_memory', True)
        self.set_default_setting('safety_checker', True)
        self.set_default_setting('float32', False if torch.cuda.is_available() else True)
        self.set_default_setting('collection', 'outputs')
        self.set_default_setting('type', 'image')
        self.set_default_setting('scheduler', 'k_euler_a')
        self.set_default_setting('model', 'runwayml/stable-diffusion-v1-5')
        self.set_default_setting('prompt', '')
        self.set_default_setting('negative_prompt', '')
        self.set_default_setting('manual_seed', False)
        self.set_default_setting('seed', 1)
        self.set_default_setting('num_images_per_prompt', 1)
        self.set_default_setting('num_inference_steps', 30)
        self.set_default_setting('guidance_scale', 7.0)
        self.set_default_setting('width', 512)
        self.set_default_setting('height', 512)
        self.set_default_setting('img2img_enabled', False)
        self.set_default_setting('img2img_source', '')
        self.set_default_setting('img2img_strength', 0.5)
        self.set_default_setting('control_net_enabled', False)
        self.set_default_setting('control_net_guidance_start', 0.0)
        self.set_default_setting('control_net_guidance_end', 1.0)
        self.set_default_setting('control_nets', '[]')
        self.set_default_setting('upscale_enabled', False)
        self.set_default_setting('upscale_factor', 2)
        self.set_default_setting('upscale_denoising_strength', 0.75)
        self.set_default_setting('upscale_blend_strength', 0.75)
        self.set_default_setting('face_enabled', False)
        self.set_default_setting('face_blend_strength', 0.75)
        self.set_default_setting('high_res_enabled', False)
        self.set_default_setting('high_res_factor', 1.5)
        self.set_default_setting('high_res_guidance_scale', 7.0)
        self.set_default_setting('high_res_noise', 0.5)
        self.set_default_setting('high_res_steps', 30)

        self.set_default_setting('huggingface_controlnet10', False)
        self.set_default_setting('huggingface_controlnet11', True)
        self.set_default_setting('huggingface_models', ['runwayml/stable-diffusion-v1-5'])

        configuration.load_from_settings(self.settings)

        # Collections
        self.collections = sorted([entry for entry in os.listdir(configuration.IMAGES_PATH) if os.path.isdir(os.path.join(configuration.IMAGES_PATH, entry))])
        if not self.collections:
            os.makedirs(os.path.join(configuration.IMAGES_PATH, 'outputs'))
            self.collections = ['outputs']

        # QT configuration
        self.setWindowIcon(QIcon(configuration.get_resource_path('app_icon.png')))
        self.setApplicationName(configuration.APP_NAME)
        qdarktheme.setup_theme('auto', corner_shape='sharp', additional_qss='QToolTip { border: 0px; }')
        fa.load()

        # Main window
        self.main_window = MainWindow(self.settings, self.collections)
        self.main_window.show()

    def set_default_setting(self, key: str, value):
        if not self.settings.contains(key):
            self.settings.setValue(key, value)

    def event(self, event):
        if event.type() == QEvent.Quit:
            if self.main_window.hide_if_thread_running():
                return False
        return super().event(event)
