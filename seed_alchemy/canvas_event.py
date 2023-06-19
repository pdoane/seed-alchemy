from dataclasses import dataclass
from PySide6.QtCore import QPointF


@dataclass
class CanvasMouseEvent:
    scene_pos: QPointF
