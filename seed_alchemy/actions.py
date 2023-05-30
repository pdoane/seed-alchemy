import sys
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QFont
from PySide6.QtWidgets import QPushButton

from . import font_awesome as fa
from . import utils
from .icon_engine import FontAwesomeIconEngine, PixmapIconEngine
from .widgets import ToolButton


@dataclass
class ActionDef:
    text: str
    icon: str = None
    fa_icon: str = None
    shortcut: QKeySequence.StandardKey = None
    checkable: bool = False
    auto_exclusive: bool = False
    empty_icon: bool = True
    role: QAction.MenuRole = None

    def create(self, parent: QObject = None):
        if self.fa_icon is not None:
            action = QAction(QIcon(FontAwesomeIconEngine(self.fa_icon)), self.text, parent)
        elif self.icon is not None:
            action = QAction(QIcon(PixmapIconEngine(self.icon)), self.text, parent)
        elif self.empty_icon:
            action = QAction(utils.empty_qicon(), self.text, parent)
        else:
            action = QAction(self.text, parent)

        if self.shortcut is not None:
            action.setShortcut(self.shortcut)
        if self.role is not None:
            action.setMenuRole(self.role)
        return action

    def tool_button(self):
        return self._tool_button(fa.font)

    def mode_button(self):
        button = self._tool_button(fa.mode_font)
        button.setFixedSize(40, 40)
        return button

    def push_button(self):
        button = QPushButton()
        if self.fa_icon is not None:
            button.setText(self.fa_icon)
            button.setFont(fa.font)
            button.setToolTip(self.text)
            button.setToolTipDuration(0)
        elif self.icon is not None:
            button.setIcon(QIcon(PixmapIconEngine(self.icon)))
            button.setToolTip(self.text)
            button.setToolTipDuration(0)
        else:
            button.setText(self.text)
        button.setCheckable(self.checkable)
        button.setAutoExclusive(self.auto_exclusive)
        return button

    def _tool_button(self, font):
        button = ToolButton()
        if self.fa_icon is not None:
            button.setText(self.fa_icon)
            button.setFont(font)
            button.setToolTip(self.text)
            button.setToolTipDuration(0)
        elif self.icon is not None:
            button.setIcon(QIcon(PixmapIconEngine(self.icon)))
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            button.setToolTip(self.text)
            button.setToolTipDuration(0)
        else:
            button.setText(self.text)
        button.setCheckable(self.checkable)
        button.setAutoExclusive(self.auto_exclusive)
        return button


# Modes
image_mode = ActionDef("Image Generation", fa_icon=fa.icon_image, checkable=True, auto_exclusive=True)
prompt_mode = ActionDef("Prompt Generation", fa_icon=fa.icon_comments, checkable=True, auto_exclusive=True)
interrogate_mode = ActionDef("Interrogate Image", fa_icon=fa.icon_question, checkable=True, auto_exclusive=True)

# File
preferences = ActionDef("Preferences...", empty_icon=False, role=QAction.MenuRole.PreferencesRole)
quit = ActionDef("Exit", empty_icon=False, role=QAction.MenuRole.QuitRole)

# History
back = ActionDef("Back", fa_icon=fa.icon_arrow_left, shortcut=Qt.CTRL | Qt.Key_BracketLeft)
forward = ActionDef("Forward", fa_icon=fa.icon_arrow_right, shortcut=Qt.CTRL | Qt.Key_BracketRight)

# Prompt
insert_textual_inversion = ActionDef("Insert Textual Inversion...")
insert_lora = ActionDef("Insert LoRA...")

set_as_image_prompt = ActionDef("Set as Image Prompt", fa_icon=fa.icon_share)

# Image
generate_image = ActionDef("Generate Image", shortcut=Qt.CTRL | Qt.Key_Return)
cancel_generation = ActionDef("Cancel Generation", fa_icon=fa.icon_ban, shortcut=Qt.SHIFT | Qt.Key_X)

locate_source = ActionDef("Locate Source Image", fa_icon=fa.icon_compass)

set_as_source_image = ActionDef("Set as Source Image", fa_icon=fa.icon_share)
use_prompt = ActionDef("Use Prompt", fa_icon=fa.icon_quote_left, shortcut=Qt.Key_P)
use_seed = ActionDef("Use Seed", fa_icon=fa.icon_seedling, shortcut=Qt.Key_S)
use_source_images = ActionDef("Use Source Images", fa_icon=fa.icon_image)
use_all = ActionDef("Use All", fa_icon=fa.icon_star_of_life, shortcut=Qt.Key_A)
toggle_metadata = ActionDef("Toggle Metadata", fa_icon=fa.icon_circle_info, shortcut=Qt.Key_I, checkable=True)
toggle_preview = ActionDef("Toggle Preview", fa_icon=fa.icon_magnifying_glass, checkable=True)

delete_image = ActionDef("Delete Image", fa_icon=fa.icon_trash, shortcut=Qt.CTRL | Qt.Key_Backspace)
if sys.platform == "darwin":
    reveal_in_finder = ActionDef("Reveal in Finder", shortcut=Qt.CTRL | Qt.ALT | Qt.Key_R)
elif sys.platform == "win32":
    reveal_in_finder = ActionDef("Reveal in File Explorer", shortcut=Qt.SHIFT | Qt.ALT | Qt.Key_R)

interrogate = ActionDef("Interrogate CLIP")

# Tools
convert_model = ActionDef("Convert .ckpt/.safetensors model...")

# Help
about = ActionDef("About", empty_icon=False, role=QAction.MenuRole.AboutRole)
