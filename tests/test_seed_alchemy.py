import os
import shutil
import threading
import time

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from seed_alchemy.application import Application
from seed_alchemy.image_mode import ImageModeWidget
from seed_alchemy.image_generation_panel import ImageGenerationPanel


def wait_for_task(task, timeout=5):
    event = threading.Event()
    task.completed.connect(lambda: event.set())

    start_time = time.time()
    while not event.is_set():
        if time.time() - start_time > timeout:
            pytest.fail(f"Timeout occurred while waiting for thread")
        QApplication.instance().processEvents()


def compare_images(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    rmse = np.sqrt(((img1_np - img2_np) ** 2).mean())
    return rmse


def set_resources_path(path):
    global resources_path
    resources_path = path


def get_resource_path(relative_path) -> str:
    return os.path.join(resources_path, relative_path)


@pytest.fixture(scope="module")
def app(request):
    app = None
    dir_name = os.path.abspath("test_data")
    set_resources_path(os.path.join(os.getcwd(), "tests", "resources"))
    try:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        app = Application(["seed_alchemy", "--root=" + dir_name])
        yield app
        app.main_window.close()
        app.exit()
    finally:
        if not request.config.test_failed:
            shutil.rmtree(dir_name)


def test(app: Application):
    main_window = app.main_window
    assert main_window is not None

    image_mode_widget: ImageModeWidget = main_window.set_mode("image")
    generation_panel = image_mode_widget.generation_panel

    generation_panel.prompt_edit.setPlainText("a fantasy landscape")
    generation_panel.manual_seed_group_box.setChecked(True)
    generation_panel.seed_line_edit.setText("2")

    generation_panel.generate_button.click()
    generate_task = image_mode_widget.generate_task
    assert generate_task is not None

    wait_for_task(generate_task, timeout=60)

    rmse = compare_images("images/outputs/00001.png", get_resource_path("00001.png"))
    assert rmse <= 0.02
