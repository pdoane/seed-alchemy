from dataclasses import dataclass

SELECTION_TOOL = 1
BRUSH_TOOL = 2
ERASER_TOOL = 3


@dataclass
class CanvasState:
    tool: int = SELECTION_TOOL


def is_paint_tool(tool):
    return tool != SELECTION_TOOL
