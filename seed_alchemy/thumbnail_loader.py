import os
from collections import deque

from PIL import Image, ImageQt
from PySide6.QtCore import QMutex, QMutexLocker, QObject, QRunnable, Qt, QThreadPool, QWaitCondition, Signal
from PySide6.QtGui import QPixmap

from . import configuration, utils


class ThumbnailRunnable(QObject, QRunnable):
    thumbnail_loaded = Signal(str, QPixmap)

    def __init__(self, loader):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self.loader = loader

    def run(self):
        while not self.loader.is_shutting_down:
            with QMutexLocker(self.loader.mutex):
                if not self.loader.requests:
                    self.loader.wait_condition.wait(self.loader.mutex)
                    if self.loader.is_shutting_down:
                        break
                    if not self.loader.requests:
                        continue
                image_path, max_size, callback = self.loader.requests.pop()

            self.process(image_path, max_size, callback)

    def process(self, image_path, max_size, callback):
        try:
            full_path = os.path.join(configuration.IMAGES_PATH, image_path)
            with Image.open(full_path) as image:
                image = utils.create_thumbnail(image, max_size)
                pixmap = ImageQt.toqpixmap(image)

        except (IOError, OSError):
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.transparent)

        self.thumbnail_loaded.connect(callback)
        self.thumbnail_loaded.emit(image_path, pixmap)
        self.thumbnail_loaded.disconnect(callback)


class ThumbnailLoader(QObject):
    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.requests = deque()
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.is_shutting_down = False

        num_runnables = min(4, self.thread_pool.maxThreadCount())
        for _ in range(num_runnables):
            runnable = ThumbnailRunnable(self)
            self.thread_pool.start(runnable)

    def shutdown(self):
        with QMutexLocker(self.mutex):
            self.is_shutting_down = True
            self.wait_condition.wakeAll()
        self.thread_pool.waitForDone()

    def get(self, image_path, max_size, callback):
        with QMutexLocker(self.mutex):
            self.requests.append((image_path, max_size, callback))
            self.wait_condition.wakeOne()
