from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QStyle,
                               QScrollArea, QSlider, QSpinBox,
                               QStyleOptionSlider, QWidget)


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
        size.setWidth(size.width() + self.verticalScrollBar().width() + self.widget().contentsMargins().right())
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
    def __init__(self, name, initial_value, parent=None, minimum=0.0, maximum=1.0):
        super().__init__(parent)

        self.minimum = minimum
        self.maximum = maximum

        label = QLabel(name)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = Slider(Qt.Horizontal)
        style_option = QStyleOptionSlider()
        self.slider.initStyleOption(style_option)
        thumb_size = self.slider.style().pixelMetric(QStyle.PM_SliderLength, style_option, self.slider)
        self.slider.setRange(0, 100)
        self.slider.setValue(self._to_slider_value(initial_value))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setFixedWidth(101 + thumb_size)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_box = DoubleSpinBox()
        self.spin_box.setAlignment(Qt.AlignCenter)
        self.spin_box.setFixedWidth(60)
        self.spin_box.setRange(minimum, maximum)
        self.spin_box.setSingleStep(0.01)
        self.spin_box.setDecimals(2)
        self.spin_box.setValue(initial_value)
        self.spin_box.valueChanged.connect(self.on_spin_box_changed)

        hlayout = QHBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(label)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spin_box)

    def on_check_box_changed(self, state):
        self.slider.setEnabled(state)
        self.spin_box.setEnabled(state)

    def on_slider_changed(self, value):
        self.spin_box.setValue(self._from_slider_value(value))

    def on_spin_box_changed(self, value):
        self.slider.setValue(self._to_slider_value(value))

    def _to_slider_value(self, value):
        range = self.maximum - self.minimum
        return round((value - self.minimum) * 100.0 / range)
    
    def _from_slider_value(self, value):
        range = self.maximum - self.minimum
        return self.minimum + value * range / 100.0
