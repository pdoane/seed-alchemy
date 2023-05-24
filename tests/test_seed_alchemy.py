import os
import threading
import time
import shutil

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


@pytest.fixture(scope="module")
def app(request):
    app = None
    dir_name = os.path.abspath("test_data")
    try:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        app = Application(["seed_alchemy", "--root=" + dir_name])
        yield app
        app.main_window.close()
    finally:
        if not request.config.test_failed:
            shutil.rmtree(dir_name)


def test(app: Application):
    main_window = app.main_window
    assert main_window is not None

    main_window.generate_button.click()
    generate_thread = main_window.generate_thread
    assert generate_thread is not None

    wait_for_thread(generate_thread, timeout=60)
