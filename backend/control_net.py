from PIL import Image

from . import control_net_registry


class ControlNetProcessor:
    def __init__(self) -> None:
        self.detector_cls = None
        self.detector = None

    def __call__(
        self, image: Image.Image, resolution: int, processor_name: str, params: dict[str, float]
    ) -> Image.Image:
        info = control_net_registry.processors[processor_name]

        if self.detector_cls != info.cls:
            self.detector = None

        if not self.detector:
            if info.repo_id:
                self.detector = info.cls.from_pretrained(info.repo_id)
            else:
                self.detector = info.cls()

            self.detector_cls = info.cls

        post_process = info.post_process or (lambda x: x)
        return post_process(
            self.detector(image, detect_resolution=resolution, image_resolution=resolution, **info.params, **params)
        )
