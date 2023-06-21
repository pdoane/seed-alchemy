import argparse
import os
import sys

import qdarktheme
import torch
from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from . import configuration, control_net_config
from . import font_awesome as fa
from .main_window import MainWindow
from .prompt_mode import PromptMetadata


class Application(QApplication):
    settings: QSettings = None
    collections: list[str] = []

    def __init__(self, argv):
        if sys.platform == "darwin":
            from Foundation import NSBundle

            bundle = NSBundle.mainBundle()
            info_dict = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            info_dict["CFBundleName"] = configuration.APP_NAME

        super().__init__(argv)

        parser = argparse.ArgumentParser(description=configuration.APP_NAME)
        parser.add_argument("--root")
        args = parser.parse_args(argv[1:])

        configuration.set_resources_path(os.path.join(os.getcwd(), "seed_alchemy/resources"))
        if args.root:
            os.chdir(os.path.expanduser(args.root))

        # Directories
        os.makedirs(configuration.IMAGES_PATH, exist_ok=True)
        os.makedirs(configuration.MODELS_PATH, exist_ok=True)
        os.makedirs(os.path.join(configuration.IMAGES_PATH, configuration.CANVAS_DIR), exist_ok=True)

        # Settings
        self.settings = QSettings("settings.ini", QSettings.IniFormat)
        self.set_default_setting("local_models_path", "")
        self.set_default_setting("reduce_memory", True)
        self.set_default_setting("safety_checker", True)
        self.set_default_setting("float32", not torch.cuda.is_available())
        self.set_default_setting("collection", "outputs")
        self.set_default_setting("mode", "image")
        self.set_default_setting("install_control_net_v10", "False")
        self.set_default_setting("install_control_net_v11", "True")
        self.set_default_setting("install_control_net_mediapipe_v2", "True")
        self.set_default_setting("huggingface_models", ["runwayml/stable-diffusion-v1-5"])
        self.set_default_setting("huggingface_promptgen", ["AUTOMATIC/promptgen-lexart"])

        prompt_meta = PromptMetadata()

        self.settings.beginGroup("promptgen")
        self.set_default_setting("model", prompt_meta.model)
        self.set_default_setting("prompt", prompt_meta.prompt)
        self.set_default_setting("count", prompt_meta.count)
        self.set_default_setting("temperature", prompt_meta.temperature)
        self.set_default_setting("top_k", prompt_meta.top_k)
        self.set_default_setting("top_p", prompt_meta.top_p)
        self.set_default_setting("num_beams", prompt_meta.num_beams)
        self.set_default_setting("repetition_penalty", prompt_meta.repetition_penalty)
        self.set_default_setting("length_penalty", prompt_meta.length_penalty)
        self.set_default_setting("min_length", prompt_meta.min_length)
        self.set_default_setting("max_length", prompt_meta.max_length)
        self.set_default_setting("seed", prompt_meta.seed)
        self.set_default_setting("manual_seed", False)
        self.settings.endGroup()

        self.settings.beginGroup("interrogate")
        self.set_default_setting("source", "")
        self.set_default_setting("caption_model", "blip-large")
        self.set_default_setting("clip_model", "ViT-L-14/openai")
        self.set_default_setting("mode", "Best")
        self.set_default_setting("max_caption_length", 32)
        self.set_default_setting("min_flavors", 8)
        self.set_default_setting("max_flavors", 32)
        self.set_default_setting("caption", "")
        self.set_default_setting("manual_caption", False)
        self.settings.endGroup()

        configuration.load_from_settings(self.settings)
        control_net_config.load_from_settings(self.settings)

        # Collections
        self.collections = sorted(
            [
                entry
                for entry in os.listdir(configuration.IMAGES_PATH)
                if entry[0] != "." and os.path.isdir(os.path.join(configuration.IMAGES_PATH, entry))
            ]
        )
        if not self.collections:
            os.makedirs(os.path.join(configuration.IMAGES_PATH, "outputs"))
            self.collections = ["outputs"]

        # QT configuration
        self.setWindowIcon(QIcon(configuration.get_resource_path("app_icon.png")))
        self.setApplicationName(configuration.APP_NAME)
        qdarktheme.setup_theme("auto", corner_shape="sharp", additional_qss="QToolTip { border: 0px; }")
        fa.load()

        # Main window
        self.main_window = MainWindow(self.settings, self.collections)
        self.main_window.show()

    def set_default_setting(self, key: str, value):
        if not self.settings.contains(key):
            self.settings.setValue(key, value)
