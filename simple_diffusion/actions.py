from dataclasses import dataclass

import configuration
import utils
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import QToolButton

@dataclass
class ActionDef:
    text: str
    icon: str = None
    shortcut: QKeySequence.StandardKey = None
    checkable: bool = False
    auto_exclusive: bool = False
    empty_icon: bool = True
    role: QAction.MenuRole = None

    def create(self, parent: QObject = None):
        if self.icon is not None:
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

# Application
about = ActionDef('About', empty_icon=False)
preferences = ActionDef('Preferences', empty_icon=False, role=QAction.MenuRole.PreferencesRole)

# Modes
txt2img = ActionDef('Text to Image', icon='txt2img_icon.png', checkable=True, auto_exclusive=True)
img2img = ActionDef('Image to Image', icon='img2img_icon.png', checkable=True, auto_exclusive=True)

# Image
generate_image = ActionDef('Generate Image', shortcut=Qt.CTRL | Qt.Key_Return)
cancel_generation = ActionDef('Cancel Generation', shortcut=Qt.SHIFT | Qt.Key_X)

locate_source = ActionDef('Locate Source Image', icon='locate_icon.png')

send_to_img2img = ActionDef('Send to Image to Image', icon='share_icon.png')
use_prompt = ActionDef('Use Prompt', icon='use_prompt_icon.png', shortcut=Qt.Key_P)
use_seed = ActionDef('Use Seed', icon='use_seed_icon.png', shortcut=Qt.Key_S)
use_initial_image = ActionDef('Use Initial Image', icon='use_initial_image_icon.png')
use_all = ActionDef('Use All', icon='use_all_icon.png', shortcut=Qt.Key_A)
toggle_metadata = ActionDef('Toggle Metadata', icon='metadata_icon.png', shortcut=Qt.Key_I, checkable=True)
toggle_preview = ActionDef('Toggle Preview', icon='preview_icon.png', checkable=True)

delete_image = ActionDef('Delete Image', icon='delete_icon.png', shortcut=Qt.Key_Delete)
reveal_in_finder = ActionDef('Reveal in Finder', shortcut=Qt.CTRL | Qt.ALT | Qt.Key_R)
