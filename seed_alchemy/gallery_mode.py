import glob
import os
import random

from PySide6.QtCore import Property, QPropertyAnimation, QSettings, Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from . import configuration
from .backend import Backend


class FadingImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap_opacity = 1.0
        self._pixmap = None
        self._resized_pixmap = None

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._resized_pixmap is not None:
            painter.setOpacity(self.pixmap_opacity)
            x = (self.width() - self._resized_pixmap.width()) / 2
            y = (self.height() - self._resized_pixmap.height()) / 2
            painter.drawPixmap(x, y, self._resized_pixmap)

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._resize()

    def resizeEvent(self, event):
        self._resize()

    def _resize(self):
        if self._pixmap:
            self._resized_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def get_pixmap_opacity(self):
        return self._pixmap_opacity

    def set_pixmap_opacity(self, value):
        self._pixmap_opacity = value
        self.update()

    pixmap_opacity = Property(float, get_pixmap_opacity, set_pixmap_opacity)


class GalleryModeWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.backend: Backend = main_window.backend
        self.settings: QSettings = main_window.settings

        collection = self.settings.value("collection")
        self.image_files = glob.glob(os.path.join(configuration.IMAGES_PATH, collection, "*"))
        random.shuffle(self.image_files)
        self.index = -1

        self.image1 = FadingImage(self)
        self.image2 = FadingImage(self)

        self.animation1 = QPropertyAnimation(self.image1, b"pixmap_opacity")
        self.animation2 = QPropertyAnimation(self.image2, b"pixmap_opacity")

        self.image1.setGeometry(self.rect())
        self.image2.setGeometry(self.rect())

        self.interval = 1200
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_next_image)
        self.auto_next_image = True

        self.set_next_image(1)

    def get_menus(self):
        return []

    def on_close(self):
        return True

    def on_key_press(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self.set_next_image(-1)
        elif key == Qt.Key_Right:
            self.set_next_image(1)
        elif key == Qt.Key_Space:
            self.auto_next_image = not self.auto_next_image
            if self.auto_next_image:
                self.timer.start(self.interval)
            else:
                self.timer.stop()
        else:
            return False
        return True

    def resizeEvent(self, event):
        self.image1.setGeometry(self.rect())
        self.image2.setGeometry(self.rect())

    def showEvent(self, event):
        if self.auto_next_image:
            self.timer.start(self.interval)

    def hideEvent(self, event):
        self.timer.stop()

    def timer_next_image(self):
        self.index = (self.index + 1) % len(self.image_files)
        image = QImage(self.image_files[self.index])
        if self.index % 2 == 0:
            self.image1.setPixmap(QPixmap.fromImage(image))
            self.start_animation(self.animation1, self.animation2)
        else:
            self.image2.setPixmap(QPixmap.fromImage(image))
            self.start_animation(self.animation2, self.animation1)

    def set_next_image(self, increment):
        if not self.image_files:
            return
        
        if self.auto_next_image:
            self.timer.stop()
            self.timer.start(self.interval)

        self.animation1.stop()
        self.animation2.stop()

        self.index = (self.index + increment) % len(self.image_files)
        image = QImage(self.image_files[self.index])
        if self.index % 2 == 0:
            self.image1.setPixmap(QPixmap.fromImage(image))
            self.image1.set_pixmap_opacity(1.0)
            self.image2.set_pixmap_opacity(0.0)
        else:
            self.image2.setPixmap(QPixmap.fromImage(image))
            self.image2.set_pixmap_opacity(1.0)
            self.image1.set_pixmap_opacity(0.0)

    def start_animation(self, fade_in, fade_out):
        fade_interval = 250

        fade_in.stop()
        fade_out.stop()

        fade_in.setDuration(fade_interval)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)

        fade_out.setDuration(fade_interval)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)

        fade_in.start()
        fade_out.start()
