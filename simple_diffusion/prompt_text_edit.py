import os
import re

import configuration
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit
from spellchecker import SpellChecker


class PromptTextEdit(QPlainTextEdit):
    return_pressed = Signal()

    def __init__(self, desired_lines, placeholder_text, parent=None):
        super().__init__(parent)
        self.spell_checker = SpellChecker()

        custom_words = ['3d']
        for entry in configuration.known_embeddings:
            name, _ = os.path.splitext(entry)
            custom_words.append(name)

        self.spell_checker.word_frequency.load_words(custom_words)

        self.word_pattern = re.compile(r'\b(?:\w+(?:-\w+)*)\b')

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
            if not self.is_known_word(word):
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

    def is_known_word(self, word):
        if self.spell_checker.known([word]):
            return True
        components = word.split('-')
        if all(self.spell_checker.known([component]) or component.isdigit() for component in components):
            return True
        return False
