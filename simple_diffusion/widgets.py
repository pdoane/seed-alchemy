
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
                               QLabel, QScrollArea, QSizePolicy, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)


class ComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.clearFocus()
        else:
            super().keyPressEvent(event)

class ScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(size.width() + self.verticalScrollBar().width())
        return size
    
class Slider(QSlider):
    def wheelEvent(self, event):
        event.ignore()

class SpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.clearFocus()
        else:
            super().keyPressEvent(event)

class FloatSliderSpinBox(QWidget):
    def __init__(self, name, initial_value, parent=None):
        super().__init__(parent)

        label = QLabel(name)
        label.setAlignment(Qt.AlignCenter)
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider = Slider(Qt.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(initial_value * 100)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_box = DoubleSpinBox()
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
        vlayout.addWidget(label)
        vlayout.addWidget(frame)

    def on_check_box_changed(self, state):
        self.slider.setEnabled(state)
        self.spin_box.setEnabled(state)

    def on_slider_changed(self, value):
        decimal_value = value / 100
        self.spin_box.setValue(decimal_value)

    def on_spin_box_changed(self, value):
        slider_value = round(value * 100)
        self.slider.setValue(slider_value)