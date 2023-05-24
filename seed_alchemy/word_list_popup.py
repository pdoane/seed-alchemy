from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsBlurEffect,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class WordListItem(QListWidgetItem):
    def __init__(self, word, matched_text, parent=None):
        super().__init__(parent)

        self.word = word
        self.matched_text = matched_text

    def build_display(self):
        label = QLabel()
        label.setTextFormat(Qt.RichText)
        label.setText(self.get_rich_text())
        return label

    def get_rich_text(self):
        index = self.word.lower().find(self.matched_text.lower())
        before = self.word[:index]
        match = self.word[index : index + len(self.matched_text)]
        after = self.word[index + len(self.matched_text) :]

        # return f"{before}<b><font color='red'>{match}</font></b>{after}"
        return f"{before}<b>{match}</b>{after}"


class WordListWidget(QWidget):
    word_selected = Signal(str)
    valid_words: list[str] = []

    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.line_edit = QLineEdit()
        self.line_edit.installEventFilter(self)
        self.layout.addWidget(self.line_edit)

        self.list_widget = QListWidget()
        self.list_widget.setFocusPolicy(Qt.NoFocus)
        self.list_widget.itemDoubleClicked.connect(self.item_double_clicked)
        self.layout.addWidget(self.list_widget)

        self.line_edit.textChanged.connect(self.update_filter)

    def eventFilter(self, obj, event):
        if obj == self.line_edit:
            if event.type() == QEvent.KeyPress:
                if event.key() == Qt.Key_Up:
                    self.list_widget.setCurrentRow(self.list_widget.currentRow() - 1)
                    return True
                elif event.key() == Qt.Key_Down:
                    self.list_widget.setCurrentRow(self.list_widget.currentRow() + 1)
                    return True
                elif event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
                    current_item = self.list_widget.currentItem()
                    if current_item:
                        self.word_selected.emit(current_item.word)
                        self.parent().close()
                    return True
        return super().eventFilter(obj, event)

    def set_valid_words(self, valid_words):
        self.valid_words = valid_words
        self.update_filter(self.line_edit.text())

    def update_filter(self, text):
        current_item = self.list_widget.currentItem()
        previous_word = current_item.word if current_item else None

        self.list_widget.clear()

        for word in self.valid_words:
            if text.lower() in word.lower():
                item = WordListItem(word, text)
                self.list_widget.addItem(item)
                label = item.build_display()
                self.list_widget.setItemWidget(item, label)

        selected_row = False
        if previous_word:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item.word == previous_word:
                    self.list_widget.setCurrentItem(item)
                    selected_row = True
                    break

        if not selected_row:
            self.list_widget.setCurrentRow(0)

    def item_double_clicked(self, item):
        self.word_selected.emit(item.word)
        self.parent().close()


class WordListPopup(QDialog):
    word_selected = Signal(str)

    def __init__(self, title, parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)

        self.effect = QGraphicsBlurEffect()
        self.effect.setBlurRadius(10)
        self.parent().window().setGraphicsEffect(self.effect)

        label = QLabel(title)

        self.word_assist_widget = WordListWidget(self)
        self.word_assist_widget.word_selected.connect(self.handle_word_selected)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.word_assist_widget)
        self.setLayout(layout)

        self.adjustSize()

        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.escape_shortcut.activated.connect(self.close)

    def closeEvent(self, event):
        self.parent().window().setGraphicsEffect(None)
        super().closeEvent(event)

    def showEvent(self, event):
        self.word_assist_widget.line_edit.setFocus()
        super().showEvent(event)

    def handle_word_selected(self, selected_word):
        self.word_selected.emit(selected_word)

    def set_valid_words(self, valid_words):
        self.word_assist_widget.set_valid_words(valid_words)
