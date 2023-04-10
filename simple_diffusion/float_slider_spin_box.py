
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFrame, QHBoxLayout,
                               QLabel, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)


class FloatSliderSpinBox(QWidget):
    def __init__(self, name, initial_value, checkable=False, parent=None):
        super().__init__(parent)

        if checkable:
            self.check_box = QCheckBox(name)
            self.check_box.setChecked(True)
            self.check_box.stateChanged.connect(self.on_check_box_changed)
        else:
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(initial_value * 100)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setFocusPolicy(Qt.StrongFocus)
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
        if checkable:
            check_box_layout = QHBoxLayout()
            check_box_layout.setAlignment(Qt.AlignCenter)
            check_box_layout.addWidget(self.check_box)
            vlayout.addLayout(check_box_layout)
        else:
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