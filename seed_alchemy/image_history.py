from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction

from . import actions

MAX_HISTORY = 16


class ImageHistory(QObject):
    current_image_changed = Signal(str)
    history_stack = []
    history_stack_index = -1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.back_action = actions.back.create(self)
        self.back_action.setEnabled(False)
        self.back_action.triggered.connect(self.navigate_back)
        self.forward_action = actions.forward.create(self)
        self.forward_action.setEnabled(False)
        self.forward_action.triggered.connect(self.navigate_forward)

    def visit(self, path):
        if self.history_stack_index >= 0:
            current_path = self.history_stack[self.history_stack_index]
            if path == current_path:
                return

        if self.history_stack_index < len(self.history_stack) - 1:
            self.history_stack = self.history_stack[: self.history_stack_index + 1]
        self.history_stack.append(path)
        self.history_stack_index += 1

        if len(self.history_stack) > MAX_HISTORY:
            self.history_stack.pop(0)
            self.history_stack_index -= 1

        self._update_current()

    def navigate_back(self):
        if not self.navigate_back_valid():
            return
        if self.history_stack_index > 0:
            self.history_stack_index -= 1
            self._update_current()

    def navigate_back_valid(self):
        return self.history_stack_index > 0

    def navigate_forward(self):
        if not self.navigate_forward_valid():
            return
        if self.history_stack_index < len(self.history_stack) - 1:
            self.history_stack_index += 1
            self._update_current()

    def navigate_forward_valid(self):
        return self.history_stack_index < len(self.history_stack) - 1

    def go_to_index(self, index):
        if 0 <= index < len(self.history_stack):
            self.history_stack_index = index
            self._update_current()

    def populate_back_menu(self, menu):
        menu.clear()
        for i in range(self.history_stack_index - 1, -1, -1):
            menu.addAction(self._create_goto_action(i))

    def populate_forward_menu(self, menu):
        menu.clear()
        for i in range(self.history_stack_index + 1, len(self.history_stack)):
            menu.addAction(self._create_goto_action(i))

    def populate_history_menu(self, menu):
        menu.clear()
        menu.addAction(self.back_action)
        menu.addAction(self.forward_action)
        menu.addSeparator()

        for i in range(0, len(self.history_stack)):
            menu.addAction(self._create_goto_action(i))

    def _update_current(self):
        self.back_action.setEnabled(self.navigate_back_valid())
        self.forward_action.setEnabled(self.navigate_forward_valid())

        path = self.history_stack[self.history_stack_index]
        self.current_image_changed.emit(path)

    def _create_goto_action(self, i):
        path = self.history_stack[i]
        action = QAction(path, self)
        action.triggered.connect(lambda checked=False, i=i: self.go_to_index(i))
        return action
