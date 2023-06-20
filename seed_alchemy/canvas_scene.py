import math

from PySide6.QtCore import QLine, QRectF, Qt, QPointF, QRect, Signal
from PySide6.QtGui import QPainter, QPen, QTransform, QPaintEvent, QMouseEvent
from PySide6.QtWidgets import QSizePolicy, QFrame

from .canvas_element import CanvasElement, deserialize_element
from .canvas_event import CanvasMouseEvent

SELECTION_TOOL = 1
BRUSH_TOOL = 2
ERASER_TOOL = 3


def is_paint_tool(tool):
    return tool != SELECTION_TOOL


class CanvasScene(QFrame):
    element_added = Signal(CanvasElement)
    element_removed = Signal(CanvasElement)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._elements: list[CanvasElement] = []
        self._start_pos = None
        self._scale = 1.0
        self._translate = QPointF(0.0, 0.0)
        self._capture_element: CanvasElement = None
        self._hover_element: CanvasElement = None
        self._selected_elements: list[CanvasElement] = []

        self.tool = SELECTION_TOOL
        self.brush_size = 30

        self.build_transforms()

        self.setFrameShape(QFrame.Panel)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def serialize(self) -> dict:
        return {
            "scale": self._scale,
            "translate": self._translate.toTuple(),
            "tool": self.tool,
            "elements": [item.serialize() for item in self._elements],
        }

    def deserialize(self, data: dict) -> None:
        self._scale = data.get("scale", 1.0)
        self._translate = QPointF(*data.get("translate", (0, 0)))
        self.tool = data.get("tool", SELECTION_TOOL)
        self.build_transforms()

        self._elements.clear()
        for item_data in data.get("elements", []):
            element = deserialize_element(item_data.get("class"), self, item_data)
            if element is not None:
                self.add_element(element)

    def paintEvent(self, event: QPaintEvent) -> None:
        region_rect = event.region().boundingRect().toRectF()
        rect = self.rect_to_scene(region_rect)

        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setTransform(self._xform)

            self.draw_grid(painter, rect)

            for element in self._elements:
                if element.bounding_rect().intersects(rect):
                    element.draw_background(painter)

            for element in self._elements:
                if element.bounding_rect().intersects(rect):
                    element.draw_content(painter)

            for element in self._elements:
                if element.bounding_rect().intersects(rect):
                    element.draw_controls(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        canvas_event = CanvasMouseEvent(scene_pos=self.point_to_scene(event.position()))

        for element in reversed(self._elements):
            if element.bounding_rect().contains(canvas_event.scene_pos) and element.contains_point(
                canvas_event.scene_pos
            ):
                if element.mouse_press_event(canvas_event):
                    self._capture_element = element
                    self.set_selection([self._capture_element])
                    return

        self.set_selection([])
        if event.button() == Qt.LeftButton:
            self._start_pos = canvas_event.scene_pos

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        canvas_event = CanvasMouseEvent(scene_pos=self.point_to_scene(event.position()))

        if self._capture_element is not None:
            self._capture_element.mouse_move_event(canvas_event)
        elif self._start_pos is not None:
            self._translate += canvas_event.scene_pos - self._start_pos
            self.build_transforms()
            self.update()
        else:
            cursor = Qt.ArrowCursor

            new_hover_element = None
            for element in reversed(self._elements):
                if element.bounding_rect().contains(canvas_event.scene_pos) and element.contains_point(
                    canvas_event.scene_pos
                ):
                    if element.accepts_hover(canvas_event):
                        new_hover_element = element
                        break

            if self._hover_element != new_hover_element:
                if self._hover_element is not None:
                    self._hover_element.hover_leave_event()
                    self._hover_element.set_hovered(False)

                self._hover_element = new_hover_element

                if self._hover_element is not None:
                    self._hover_element.set_hovered(True)
                    self._hover_element.hover_enter_event()

            if self._hover_element is not None:
                hover_cursor = self._hover_element.hover_move_event(canvas_event)
                if hover_cursor is not None:
                    cursor = hover_cursor

            self.setCursor(cursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        canvas_event = CanvasMouseEvent(scene_pos=self.point_to_scene(event.position()))

        if self._capture_element is not None:
            self._capture_element.mouse_release_event(canvas_event)
            self._capture_element = None
        else:
            self._start_pos = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            selected = set(self._selected_elements)
            i = len(self._elements) - 1
            while i >= 0:
                element = self._elements[i]
                if element in selected:
                    self.remove_element(element)
                i -= 1

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            dy = event.angleDelta().y()
            if dy:
                scale_rate = 1.005
                scale_delta = pow(scale_rate if dy > 0 else 1 / scale_rate, abs(dy))

                self._scale *= scale_delta

                old_pos = self.point_to_scene(event.position())
                self.build_transforms()
                new_pos = self.point_to_scene(event.position())

                self._translate += new_pos - old_pos
                self.build_transforms()

                self.update()

    def leaveEvent(self, event):
        if self._hover_element:
            self._hover_element.hover_leave_event()
            self._hover_element.set_hovered(False)
            self._hover_element = None

        super().leaveEvent(event)

    def elements(self):
        return self._elements

    def add_element(self, element):
        # Insert in sorted order by layer
        insert_index = next(
            (i for i, e in enumerate(self._elements) if e.layer() > element.layer()),
            len(self._elements),
        )
        self._elements.insert(insert_index, element)
        self.update_element(element)
        self.element_added.emit(element)

    def remove_element(self, element):
        index = self._elements.index(element)
        if index != -1:
            self.remove_index(index)

    def remove_index(self, index):
        element = self._elements[index]
        self.element_removed.emit(element)
        self.update_element(element)
        self._elements.pop(index)

    def set_selection(self, selected_elements: list[CanvasElement]):
        set1 = set(self._selected_elements)
        set2 = set(selected_elements)

        for element in set1 - set2:
            element.set_selected(False)
        for element in set2 - set1:
            element.set_selected(True)

        self._selected_elements = selected_elements

    def draw_grid(self, painter: QPainter, rect: QRectF):
        left = math.floor(rect.left())
        right = math.ceil(rect.right())
        top = math.floor(rect.top())
        bottom = math.ceil(rect.bottom())

        first_left = left - (left % 64)
        first_top = top - (top % 64)

        lines = []
        for x in range(first_left, right, 64):
            lines.append(QLine(x, top, x, bottom))
        for y in range(first_top, bottom, 64):
            lines.append(QLine(left, y, right, y))

        painter.save()
        painter.setPen(QPen(Qt.darkGray, 1, Qt.SolidLine))
        painter.drawLines(lines)
        painter.restore()

    def build_transforms(self):
        self._xform = QTransform()
        self._xform.scale(self._scale, self._scale)
        self._xform.translate(self._translate.x(), self._translate.y())

        self._inv_xform = QTransform()
        self._inv_xform.translate(-self._translate.x(), -self._translate.y())
        self._inv_xform.scale(1.0 / self._scale, 1.0 / self._scale)

    def point_to_scene(self, point: QPointF) -> QPointF:
        if type(point) is not QPointF:
            raise ValueError("point must be in floating point")
        return self._inv_xform.map(point)

    def rect_to_scene(self, rect: QRectF) -> QRectF:
        if type(rect) is not QRectF:
            raise ValueError("rect must be in floating point")
        return QRectF(self._inv_xform.map(rect.topLeft()), self._inv_xform.map(rect.bottomRight()))

    def rect_to_view(self, rect: QRectF) -> QRectF:
        if type(rect) is not QRectF:
            raise ValueError("rect must be in floating point")
        return QRectF(self._xform.map(rect.topLeft()), self._xform.map(rect.bottomRight()))

    def update_scene_rect(self, rect: QRectF):
        view_rect = self.rect_to_view(rect)
        t = math.floor(view_rect.top())
        l = math.floor(view_rect.left())
        b = math.ceil(view_rect.bottom())
        r = math.ceil(view_rect.right())
        update_rect = QRect(l, t, r - l, b - t)
        self.update(update_rect)

    def update_element(self, element: CanvasElement):
        self.update_scene_rect(element.bounding_rect())
