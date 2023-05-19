from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import QPushButton, QToolButton

from . import configuration
from . import font_awesome as fa
from . import utils


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
            action = QAction(QIcon(utils.create_fontawesome_icon(self.fa_icon)), self.text, parent)
        elif self.icon is not None:
            action = QAction(QIcon(configuration.get_resource_path(self.icon)), self.text, parent)
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
        button = QToolButton()
        if self.fa_icon is not None:
            button.setText(self.fa_icon)
            button.setFont(fa.font)
            #button.setIcon(utils.create_fontawesome_icon(self.fa_icon))
            button.setToolTip(self.text)
            button.setToolTipDuration(0)
        elif self.icon is not None:
            button.setIcon(QIcon(configuration.get_resource_path(self.icon)))
            button.setIconSize(configuration.ICON_SIZE)
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            button.setToolTip(self.text)
            button.setToolTipDuration(0)
        else:
            button.setText(self.text)
        button.setCheckable(self.checkable)
        button.setAutoExclusive(self.auto_exclusive)
        return button
    
    def push_button(self):
        button = QPushButton()
        if self.fa_icon is not None:
            button.setText(self.fa_icon)
            button.setFont(fa.font)
            #button.setIcon(utils.create_fontawesome_icon(self.fa_icon))
            button.setToolTip(self.text)
            button.setToolTipDuration(0)
        elif self.icon is not None:
            button.setIcon(QIcon(configuration.get_resource_path(self.icon)))
            button.setIconSize(configuration.ICON_SIZE)
            button.setToolTip(self.text)
            button.setToolTipDuration(0)
        else:
            button.setText(self.text)
        button.setCheckable(self.checkable)
        button.setAutoExclusive(self.auto_exclusive)
        return button

# Application
about = ActionDef('About', empty_icon=False)
preferences = ActionDef('Preferences', empty_icon=False, role=QAction.MenuRole.PreferencesRole)

# Modes
image_mode = ActionDef('Image Generation', icon='img2img_icon.png', checkable=True, auto_exclusive=True)

# History
back = ActionDef('Back', fa_icon=fa.icon_arrow_left, shortcut=Qt.CTRL | Qt.Key_BracketLeft)
forward = ActionDef('Forward', fa_icon=fa.icon_arrow_right, shortcut=Qt.CTRL | Qt.Key_BracketRight)

# Image
generate_image = ActionDef('Generate Image', shortcut=Qt.CTRL | Qt.Key_Return)
cancel_generation = ActionDef('Cancel Generation', fa_icon=fa.icon_ban, shortcut=Qt.SHIFT | Qt.Key_X)

locate_source = ActionDef('Locate Source Image', fa_icon=fa.icon_compass)

set_as_source_image = ActionDef('Set as Source Image', fa_icon=fa.icon_share)
use_prompt = ActionDef('Use Prompt', fa_icon=fa.icon_quote_left, shortcut=Qt.Key_P)
use_seed = ActionDef('Use Seed', fa_icon=fa.icon_seedling, shortcut=Qt.Key_S)
use_source_images = ActionDef('Use Source Images', fa_icon=fa.icon_image)
use_all = ActionDef('Use All', fa_icon=fa.icon_star_of_life, shortcut=Qt.Key_A)
toggle_metadata = ActionDef('Toggle Metadata', fa_icon=fa.icon_circle_info, shortcut=Qt.Key_I, checkable=True)
toggle_preview = ActionDef('Toggle Preview', fa_icon=fa.icon_magnifying_glass, checkable=True)

delete_image = ActionDef('Delete Image', fa_icon=fa.icon_trash, shortcut=Qt.CTRL | Qt.Key_Backspace)
reveal_in_finder = ActionDef('Reveal in Finder', shortcut=Qt.CTRL | Qt.ALT | Qt.Key_R)
