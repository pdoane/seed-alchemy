from dataclasses import dataclass

import configuration
import font_awesome as fa
import utils
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import QToolButton, QPushButton


@dataclass
class ActionDef:
    text: str
    icon: str = None
    shortcut: QKeySequence.StandardKey = None
    checkable: bool = False
    auto_exclusive: bool = False
    empty_icon: bool = True
    use_fa: bool = False
    role: QAction.MenuRole = None

    def create(self, parent: QObject = None):
        if self.icon is not None:
            if self.use_fa:
                action = QAction(QIcon(utils.create_fontawesome_icon(self.icon)), self.text, parent)
            else:
                action = QAction(QIcon(utils.resource_path(self.icon)), self.text, parent)
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
        if self.icon is not None:
            if self.use_fa:
                button.setText(self.icon)
                button.setFont(fa.font)
                #button.setIcon(utils.create_fontawesome_icon(self.icon))
            else:
                button.setIcon(QIcon(utils.resource_path(self.icon)))
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
        if self.icon is not None:
            if self.use_fa:
                button.setText(self.icon)
                button.setFont(fa.font)
                #button.setIcon(utils.create_fontawesome_icon(self.icon))
            else:
                button.setIcon(QIcon(utils.resource_path(self.icon)))
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

# Image
generate_image = ActionDef('Generate Image', shortcut=Qt.CTRL | Qt.Key_Return)
cancel_generation = ActionDef('Cancel Generation', icon=fa.icon_ban, use_fa=True, shortcut=Qt.SHIFT | Qt.Key_X)

locate_source = ActionDef('Locate Source Image', icon=fa.icon_compass, use_fa=True)

send_to_img2img = ActionDef('Send to Image to Image', icon=fa.icon_share, use_fa=True)
use_prompt = ActionDef('Use Prompt', icon=fa.icon_quote_left, use_fa=True, shortcut=Qt.Key_P)
use_seed = ActionDef('Use Seed', icon=fa.icon_seedling, use_fa=True, shortcut=Qt.Key_S)
use_initial_image = ActionDef('Use Initial Image', icon=fa.icon_image, use_fa=True)
use_all = ActionDef('Use All', icon=fa.icon_star_of_life, use_fa=True, shortcut=Qt.Key_A)
toggle_metadata = ActionDef('Toggle Metadata', icon=fa.icon_circle_info, use_fa=True, shortcut=Qt.Key_I, checkable=True)
toggle_preview = ActionDef('Toggle Preview', icon=fa.icon_magnifying_glass, use_fa=True, checkable=True)

delete_image = ActionDef('Delete Image', icon=fa.icon_trash, use_fa=True, shortcut=Qt.Key_Delete)
reveal_in_finder = ActionDef('Reveal in Finder', shortcut=Qt.CTRL | Qt.ALT | Qt.Key_R)
