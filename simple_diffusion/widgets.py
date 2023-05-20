from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QScrollArea, QSlider, QSpinBox, QStyle,
                               QStyleOptionSlider, QToolButton, QWidget)


def round_to_step(num, step):
    return round(num / step) * step

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

class ToolButton(QToolButton):
    about_to_show_menu = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if self.popupMode() == QToolButton.InstantPopup:
            self.about_to_show_menu.emit()
        super().mousePressEvent(event)

class FloatSliderSpinBox(QWidget):
    def __init__(self, name, initial_value, parent=None, minimum=0.0, maximum=1.0, single_step=0.01):
        super().__init__(parent)

        self.single_step = single_step

        label = QLabel(name)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = Slider(Qt.Horizontal)
        style_option = QStyleOptionSlider()
        self.slider.initStyleOption(style_option)
        thumb_size = self.slider.style().pixelMetric(QStyle.PM_SliderLength, style_option, self.slider)
        self.slider.setRange(int(minimum * 100.0), int(maximum * 100.0))
        self.slider.setValue(self._to_slider_value(initial_value))
        self.slider.setSingleStep(int(single_step * 100.0))
        self.slider.setPageStep((int(maximum - minimum) * 10.0))
        self.slider.setFixedWidth(101 + thumb_size)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_box = DoubleSpinBox()
        self.spin_box.setAlignment(Qt.AlignCenter)
        self.spin_box.setFixedWidth(60)
        self.spin_box.setRange(minimum, maximum)
        self.spin_box.setSingleStep(single_step)
        self.spin_box.setDecimals(2)
        self.spin_box.setValue(initial_value)
        self.spin_box.valueChanged.connect(self.on_spin_box_changed)

        hlayout = QHBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(label)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spin_box)

    def on_slider_changed(self, value):
        self.spin_box.setValue(self._from_slider_value(value))

    def on_spin_box_changed(self, value):
        self.slider.setValue(self._to_slider_value(value))

    def _to_slider_value(self, value):
        return round(value * 100.0)
    
    def _from_slider_value(self, value):
        return round_to_step(value / 100.0, self.single_step)

class IntSliderSpinBox(QWidget):
    def __init__(self, name, initial_value, parent=None, minimum=0, maximum=100):
        super().__init__(parent)

        label = QLabel(name)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = Slider(Qt.Horizontal)
        style_option = QStyleOptionSlider()
        self.slider.initStyleOption(style_option)
        thumb_size = self.slider.style().pixelMetric(QStyle.PM_SliderLength, style_option, self.slider)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(initial_value)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setFixedWidth(101 + thumb_size)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_box = SpinBox()
        self.spin_box.setAlignment(Qt.AlignCenter)
        self.spin_box.setFixedWidth(60)
        self.spin_box.setRange(minimum, maximum)
        self.spin_box.setSingleStep(1)
        self.spin_box.setValue(initial_value)
        self.spin_box.valueChanged.connect(self.on_spin_box_changed)

        hlayout = QHBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(label)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spin_box)

    def on_slider_changed(self, value):
        self.spin_box.setValue(value)

    def on_spin_box_changed(self, value):
        self.slider.setValue(value)
