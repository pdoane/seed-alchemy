import textwrap

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from . import utils
from .canvas_element import CanvasElement
from .canvas_image_element import CanvasImageElement
from .canvas_generation_element import CanvasGenerationElement
from .widgets import ElidedLabel

ICON_SIZE = 32


class CanvasLayerWidget(QWidget):
    def __init__(self, element: CanvasElement, parent=None):
        super().__init__(parent)
        self.element = element

        self.icon: QLabel = None
        self.label: QLabel = None
        self.init_ui()

        element.hovered_changed.connect(self.update)
        element.selection_changed.connect(self.update)

        if type(element) == CanvasGenerationElement:
            generation_element: CanvasGenerationElement = element
            pixmap = QPixmap(ICON_SIZE, ICON_SIZE)
            with QPainter(pixmap) as painter:
                painter.fillRect(pixmap.rect(), utils.checkerboard_qbrush())

            params = generation_element.params()

            self.update_image_icon(pixmap)
            self.update_label(params.get("prompt", ""), params.get("negative_prompt", ""))

        if type(element) == CanvasImageElement:
            image_element: CanvasImageElement = element
            image_metadata = image_element.metadata()

            self.update_image_icon(image_element.pixmap())
            self.update_label(image_metadata.prompt, image_metadata.negative_prompt)

            image_element.pixmap_changed.connect(self.update_image_icon)

    def init_ui(self):
        self.icon = QLabel()
        self.icon.setFrameStyle(QFrame.Box)
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setFixedSize(ICON_SIZE, ICON_SIZE)

        self.label = ElidedLabel()
        self.label.setFixedWidth(200)
        self.label.setToolTipDuration(0)

        hlayout = QHBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(2)
        hlayout.addWidget(self.icon)
        hlayout.addWidget(self.label)
        hlayout.addStretch()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        if self.element.selected():
            painter.fillRect(rect, QColor("green"))
        elif self.element.hovered():
            painter.fillRect(rect, QColor("blue"))

    def enterEvent(self, event):
        self.element.set_hovered(True)
        self.update()

    def leaveEvent(self, event):
        self.element.set_hovered(False)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.element.scene().set_selection([self.element])
        self.update()

    def update_image_icon(self, pixmap: QPixmap):
        self.icon.setPixmap(pixmap.scaledToHeight(ICON_SIZE))

    def update_label(self, prompt: str, negative_prompt: str):
        self.label.setText(prompt)
        self.label.setToolTip(
            textwrap.fill(
                "<b>Prompt</b>: {:s}<br/><br/><b>Negative Prompt</b>: {:s}".format(
                    prompt,
                    negative_prompt,
                ),
                50,
            )
        )
