import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from seed_alchemy.application import Application


def wait_for_thread(thread, timeout=5):
    event = threading.Event()
    thread.finished.connect(lambda: event.set())

    start_time = time.time()
    while not event.is_set():
        if time.time() - start_time > timeout:
            pytest.fail(f"Timeout occurred while waiting for thread")
        QApplication.instance().processEvents()

@pytest.fixture(scope='module')
def app():
    app = Application([])
    yield app
    app.main_window.close()

def test(app: Application):
    main_window = app.main_window
    assert main_window is not None

    main_window.generate_button.click()
    generate_thread = main_window.generate_thread
    assert generate_thread is not None

    wait_for_thread(generate_thread, timeout=60)
