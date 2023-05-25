import os
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QFontMetrics, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit
from spellchecker import SpellChecker

from . import configuration
from .word_list_popup import WordListPopup


class PromptTextEdit(QPlainTextEdit):
    return_pressed = Signal()
    insert_textual_inversion_action: QAction
    insert_lora_action: QAction

    def __init__(self, desired_lines, placeholder_text, parent=None):
        super().__init__(parent)

        self.insert_textual_inversion_action = QAction("Insert Textual Inversion...")
        self.insert_textual_inversion_action.triggered.connect(self.on_insert_textual_inversion)
        self.insert_lora_action = QAction("Insert LoRA...")
        self.insert_lora_action.triggered.connect(self.on_insert_lora)

        self.spell_checker = SpellChecker()

        custom_words = ["3d", "useLora", "withLora"]
        custom_words += configuration.textual_inversions.keys()
        custom_words += configuration.loras.keys()

        self.spell_checker.word_frequency.load_words(custom_words)

        self.word_pattern = re.compile(r"\b(?:\w+(?:-\w+)*)\b")

        font = self.font()
        font_metrics = QFontMetrics(font)
        line_height = font_metrics.lineSpacing()
        margins = self.contentsMargins()
        frame_width = self.frameWidth()
        document_margins = self.document().documentMargin()

        self.setFixedHeight(
            line_height * desired_lines + margins.top() + margins.bottom() + 2 * frame_width + 2 * document_margins
        )
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

    def setPlainText(self, text):
        super().setPlainText(text)
        self.highlight_misspelled_words()

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        first_action = menu.actions()[0]
        menu.insertAction(first_action, self.insert_textual_inversion_action)
        menu.insertAction(first_action, self.insert_lora_action)
        menu.insertSeparator(first_action)
        menu.exec(event.globalPos())

    def on_insert_textual_inversion(self):
        popup = WordListPopup("Insert Textual Inversion", self)
        popup.word_selected.connect(self.handle_textual_inversion)

        popup.set_valid_words(configuration.textual_inversions.keys())

        pos = self.parentWidget().mapToGlobal(self.geometry().bottomLeft())
        popup.move(pos)
        popup.show()
        popup.exec()

    def handle_textual_inversion(self, selected_word):
        cursor = self.textCursor()
        cursor.insertText(selected_word)
        self.setTextCursor(cursor)

    def on_insert_lora(self):
        popup = WordListPopup("Insert LoRA", self)
        popup.word_selected.connect(self.handle_lora)

        popup.set_valid_words(configuration.loras.keys())

        pos = self.parentWidget().mapToGlobal(self.geometry().bottomLeft())
        popup.move(pos)
        popup.show()
        popup.exec()

    def handle_lora(self, selected_word):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(" useLora({:s}, 1.0)".format(selected_word))
        self.setTextCursor(cursor)

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
        components = word.split("-")
        if all(self.spell_checker.known([component]) or component.isdigit() for component in components):
            return True
        return False
