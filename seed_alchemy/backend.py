import sys
import traceback
import gc

from PySide6.QtCore import QObject, QRunnable, Signal, QThreadPool
from PySide6.QtWidgets import QProgressBar


if sys.platform == "darwin":
    from AppKit import NSApplication


class CancelTaskException(Exception):
    pass


class BackendTask(QObject, QRunnable):
    completed = Signal()

    def __init__(self):
        QObject.__init__(self)
        QRunnable.__init__(self)

    def run(self):
        try:
            self.run_()
        except CancelTaskException:
            pass
        except Exception as e:
            traceback.print_exc()

        gc.collect()
        self.completed.emit()


class Backend:
    def __init__(self, progress_bar: QProgressBar):
        self.progress_bar = progress_bar
        self.thread_pool = QThreadPool()
        self.task_count = 0

        self.thread_pool.setMaxThreadCount(1)

    def start(self, task: BackendTask):
        if self.task_count == 0:
            self.update_progress(0, 0)

        self.task_count += 1
        task.completed.connect(self._task_complete)
        self.thread_pool.start(task)

    def _task_complete(self):
        self.task_count -= 1
        if self.task_count == 0:
            self.update_progress(None)
        else:
            self.update_progress(0, 0)

    def update_progress(self, progress_amount, maximum_amount=100):
        self.progress_bar.setMaximum(maximum_amount)
        if maximum_amount == 0:
            self.progress_bar.setStyleSheet(
                "QProgressBar { border: none; } QProgressBar:chunk { background-color: grey; }"
            )
        else:
            self.progress_bar.setStyleSheet(
                "QProgressBar { border: none; } QProgressBar:chunk { background-color: blue; }"
            )
        if progress_amount is not None:
            self.progress_bar.setValue(progress_amount)
        else:
            self.progress_bar.setValue(0)

        if sys.platform == "darwin":
            sharedApplication = NSApplication.sharedApplication()
            dockTile = sharedApplication.dockTile()
            if maximum_amount == 0:
                dockTile.setBadgeLabel_("...")
            elif progress_amount is not None:
                dockTile.setBadgeLabel_("{:d}%".format(progress_amount))
            else:
                dockTile.setBadgeLabel_(None)
