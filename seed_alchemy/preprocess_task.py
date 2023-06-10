from PIL import Image
from PySide6.QtCore import Signal

from . import configuration, control_net_config
from .backend import BackendTask


class PreprocessTask(BackendTask):
    image_completed = Signal(Image.Image)

    def __init__(self, source_image: Image.Image, preprocessor_name: str, params: list[float]):
        super().__init__()

        self.source_image = source_image
        self.preprocessor_name = preprocessor_name
        self.params = params

    def run_(self):
        preprocessor_type = control_net_config.preprocessors.get(self.preprocessor_name)
        if preprocessor_type is None:
            return

        preprocessor = preprocessor_type(configuration.torch_device)
        processed_image = preprocessor(self.source_image, self.params)
        self.image_completed.emit(processed_image)
