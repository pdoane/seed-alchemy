import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit
from spellchecker import SpellChecker


class PromptTextEdit(QPlainTextEdit):
    return_pressed = Signal()

    def __init__(self, desired_lines, placeholder_text, parent=None):
        super().__init__(parent)
        self.spell_checker = SpellChecker()
        self.word_pattern = re.compile(r'\b\w+\b')

        font = self.font()
        font.setPointSize(14)
        self.setFont(font)

        font_metrics = QFontMetrics(font)
        line_height = font_metrics.lineSpacing()
        margins = self.contentsMargins()
        frame_width = self.frameWidth()
        document_margins = self.document().documentMargin()

        self.setFixedHeight(line_height * desired_lines + margins.top() + margins.bottom() + 2 * frame_width + 2 * document_margins)
        self.setPlaceholderText(placeholder_text)
        self.setTabChangesFocus(True)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Enter, Qt.Key_Return):
            self.clearFocus()
            self.return_pressed.emit()
        elif key == Qt.Key_Escape:
            self.clearFocus()
        else:
            super().keyPressEvent(event)
            self.highlight_misspelled_words()

        if key in (Qt.Key_Left, Qt.Key_Right):
            event.accept()

    def setPlainText(self, str):
        super().setPlainText(str)
        self.highlight_misspelled_words()

    def highlight_misspelled_words(self):
        text = self.toPlainText()
        words = self.word_pattern.finditer(text)

        extra_selections = []
        for match in words:
            word = match.group()
            if not self.spell_checker.known([word]):
                format = QTextCharFormat()
                format.setUnderlineColor(QColor(Qt.red))
                format.setUnderlineStyle(QTextCharFormat.SingleUnderline)

                index = match.start()
                cursor = QTextCursor(self.document())
                cursor.setPosition(index, QTextCursor.MoveAnchor)
                cursor.setPosition(index + len(word), QTextCursor.KeepAnchor)

                extra_selection = QTextEdit.ExtraSelection()
                extra_selection.cursor = cursor
                extra_selection.format = format
                extra_selections.append(extra_selection)

        self.setExtraSelections(extra_selections)
