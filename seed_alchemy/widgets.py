from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QWidget,
)

from . import font_awesome as fa


def round_to_step(num, step):
    return round(num / step) * step


class ComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        _ = self.sizeHint()
        self.setFixedHeight(self.minimumHeight())

    def wheelEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.clearFocus()
        else:
            super().keyPressEvent(event)


class FrameWithCloseButton(QFrame):
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.close_button = None

    def showEvent(self, event):
        if self.close_button is None:
            self.close_button = QPushButton(self)
            self.close_button.setText(fa.icon_xmark)
            self.close_button.setFont(fa.font)
            self.close_button.setToolTip("Remove")
            self.close_button.setFixedSize(self.close_button.sizeHint())
            self.close_button.clicked.connect(self._emit_closed)

            self._move()
            self.close_button.show()
        super().showEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._move()

    def _move(self):
        if self.close_button:
            self.close_button.move(self.width() - self.close_button.width(), 0)

    def _emit_closed(self):
        self.closed.emit()


class ScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

    def sizeHint(self):
        self.verticalScrollBar().adjustSize()
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
        _ = self.sizeHint()
        self.setFixedHeight(self.minimumHeight())

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


class ElidedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(100)

    def paintEvent(self, event):
        painter = QPainter(self)
        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(self.text(), Qt.ElideRight, self.width())
        painter.drawText(self.rect(), self.alignment(), elided)


class FloatSliderSpinBox(QWidget):
    def __init__(self, name, minimum=0.0, maximum=1.0, step=0.01, decimals=2, parent=None):
        super().__init__(parent)

        self.step = step

        self.label = QLabel(name)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = Slider(Qt.Horizontal)
        style_option = QStyleOptionSlider()
        self.slider.initStyleOption(style_option)
        thumb_size = self.slider.style().pixelMetric(QStyle.PM_SliderLength, style_option, self.slider)
        self.slider.setRange(int(minimum * 100.0), int(maximum * 100.0))
        self.slider.setSingleStep(int(step * 100.0))
        self.slider.setPageStep(int(step * 1000.0))
        self.slider.setFixedWidth(101 + thumb_size)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_box = DoubleSpinBox()
        self.spin_box.setAlignment(Qt.AlignCenter)
        self.spin_box.setFixedWidth(60)
        self.spin_box.setRange(minimum, maximum)
        self.spin_box.setSingleStep(step)
        self.spin_box.setDecimals(decimals)
        self.spin_box.valueChanged.connect(self.on_spin_box_changed)

        hlayout = QHBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(self.label)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spin_box)

    def value(self):
        return self.spin_box.value()

    def setValue(self, val):
        self.spin_box.setValue(val)

    def on_slider_changed(self, value):
        self.spin_box.setValue(self._from_slider_value(value))

    def on_spin_box_changed(self, value):
        self.slider.setValue(self._to_slider_value(value))

    def _to_slider_value(self, value):
        return round(value * 100.0)

    def _from_slider_value(self, value):
        return round_to_step(value / 100.0, self.step)


class IntSliderSpinBox(QWidget):
    def __init__(self, name, minimum=0, maximum=100, step=1, parent=None):
        super().__init__(parent)

        self.label = QLabel(name)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = Slider(Qt.Horizontal)
        style_option = QStyleOptionSlider()
        self.slider.initStyleOption(style_option)
        thumb_size = self.slider.style().pixelMetric(QStyle.PM_SliderLength, style_option, self.slider)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(step * 10)
        self.slider.setFixedWidth(101 + thumb_size)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_box = SpinBox()
        self.spin_box.setAlignment(Qt.AlignCenter)
        self.spin_box.setFixedWidth(60)
        self.spin_box.setRange(minimum, maximum)
        self.spin_box.setSingleStep(step)
        self.spin_box.valueChanged.connect(self.on_spin_box_changed)

        hlayout = QHBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(self.label)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spin_box)

    def value(self):
        return self.spin_box.value()

    def setValue(self, val):
        self.spin_box.setValue(val)

    def on_slider_changed(self, value):
        self.spin_box.setValue(value)

    def on_spin_box_changed(self, value):
        self.slider.setValue(value)
