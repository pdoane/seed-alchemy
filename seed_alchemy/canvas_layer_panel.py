from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QGroupBox, QSizePolicy, QVBoxLayout, QPushButton, QScrollArea

from .widgets import ScrollArea
from .canvas_generation_element import CanvasGenerationElement
from .canvas_image_element import CanvasImageElement
from .canvas_layer_widget import CanvasLayerWidget
from .canvas_element import CanvasElement
from .canvas_scene import CanvasScene


class CanvasLayerPanel(ScrollArea):
    def __init__(self, scene: CanvasScene, parent=None):
        super().__init__(parent)
        self.scene = scene

        # Frame
        self.frame = QFrame()
        # self.setContentsMargins(0, 0, 2, 0)

        # Scroll Area Configuration
        self.setFrameStyle(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.setWidgetResizable(True)
        self.setWidget(self.frame)
        self.setFocusPolicy(Qt.NoFocus)

        # Generators
        self.generator_box = QGroupBox("Generators")

        self.generator_elements_layout = QVBoxLayout()
        self.generator_elements_layout.setContentsMargins(0, 0, 0, 0)

        generator_box_layout = QVBoxLayout(self.generator_box)
        generator_box_layout.addLayout(self.generator_elements_layout)

        # Images
        self.image_box = QGroupBox("Images")

        self.image_elements_layout = QVBoxLayout()
        self.image_elements_layout.setContentsMargins(0, 0, 0, 0)

        image_box_layout = QVBoxLayout(self.image_box)
        image_box_layout.addLayout(self.image_elements_layout)

        # Layout
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.generator_box)
        frame_layout.addWidget(self.image_box)
        frame_layout.addStretch()

        scene.element_added.connect(self.on_element_added)
        scene.element_removed.connect(self.on_element_removed)

    def on_element_added(self, element: CanvasElement):
        layer_widget = CanvasLayerWidget(element)
        if type(element) == CanvasGenerationElement:
            self.generator_elements_layout.insertWidget(0, layer_widget)
        else:
            self.image_elements_layout.insertWidget(0, layer_widget)

    def on_element_removed(self, element: CanvasElement):
        if type(element) == CanvasGenerationElement:
            layout = self.generator_elements_layout
        else:
            layout = self.image_elements_layout

        for i in range(layout.count()):
            widget: CanvasLayerWidget = layout.itemAt(i).widget()
            if widget.element == element:
                layout.removeWidget(widget)
                widget.setParent(None)
                break

        self.frame.adjustSize()
