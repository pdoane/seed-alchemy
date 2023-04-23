import configuration
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QFileDialog, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)


class DirectoryPathWidget(QWidget):
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label_text)
        self.line_edit = QLineEdit(self)
        self.button = QPushButton("Browse...", self)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.line_edit)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.browse)

    def browse(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(
            self, "Select a directory", "", options=options
        )
        if directory:
            self.line_edit.setText(directory)

    def path(self):
        return self.line_edit.text()

    def set_path(self, path):
        self.line_edit.setText(path)

class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Preferences")
        self.settings = QSettings("settings.ini", QSettings.IniFormat)

        restartLabel = QLabel("Changes to application settings may require a restart.")

        self.embeddings_path = DirectoryPathWidget('Embeddings Path')
        self.embeddings_path.set_path(self.settings.value('embeddings_path'))

        self.reduce_memory = QCheckBox('Reduce Memory')
        self.reduce_memory.setChecked(self.settings.value('reduce_memory', type=bool))

        self.safety_checker = QCheckBox('Safety Checker')
        self.safety_checker.setChecked(self.settings.value('safety_checker', type=bool))

        models_group = QGroupBox()

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Display Name", "Repository ID"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_model)

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_model)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.add_button)

        models_group_layout = QVBoxLayout(models_group)
        models_group_layout.addWidget(self.table)
        models_group_layout.addLayout(button_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(restartLabel)
        layout.addSpacing(8)
        layout.addWidget(self.embeddings_path)
        layout.addWidget(self.reduce_memory)
        layout.addWidget(self.safety_checker)
        layout.addWidget(models_group)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setMinimumWidth(600)

        self.load_models()

    def load_models(self):
        self.settings.beginGroup("Models")
        keys = self.settings.childKeys()
        self.table.setRowCount(len(keys))

        for i, key in enumerate(keys):
            display_name = key
            repo_id = self.settings.value(key)

            self.table.setItem(i, 0, QTableWidgetItem(display_name))
            self.table.setItem(i, 1, QTableWidgetItem(repo_id))

        self.settings.endGroup()

    def remove_model(self):
        current_row = self.table.currentRow()

        if current_row == -1:
            QMessageBox.warning(self, "Warning", "Please select a model to remove.")
            return

        self.table.removeRow(current_row)

    def add_model(self):
        add_dialog = QDialog(self)
        add_dialog.setWindowTitle("Add Model")

        vbox = QVBoxLayout()

        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("Display Name:"))
        display_name_edit = QLineEdit()
        hbox1.addWidget(display_name_edit)
        vbox.addLayout(hbox1)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Repository ID:"))
        repo_id_edit = QLineEdit()
        hbox2.addWidget(repo_id_edit)
        vbox.addLayout(hbox2)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(add_dialog.accept)
        button_box.rejected.connect(add_dialog.reject)
        vbox.addWidget(button_box)

        add_dialog.setLayout(vbox)

        result = add_dialog.exec()
        if result == QDialog.Accepted:
            display_name = display_name_edit.text()
            repo_id = repo_id_edit.text()

            if not display_name or not repo_id:
                return

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(display_name))
            self.table.setItem(row, 1, QTableWidgetItem(repo_id))

    def accept(self):
        self.settings.setValue('embeddings_path', self.embeddings_path.path())
        self.settings.setValue('reduce_memory', self.reduce_memory.isChecked())
        self.settings.setValue('safety_checker', self.safety_checker.isChecked())

        self.settings.beginGroup("Models")
        self.settings.remove("")

        for row in range(self.table.rowCount()):
            display_name = self.table.item(row, 0).text()
            repo_id = self.table.item(row, 1).text()

            self.settings.setValue(display_name, repo_id)

        self.settings.endGroup()
        configuration.load_from_settings(self.settings)

        super().accept()
