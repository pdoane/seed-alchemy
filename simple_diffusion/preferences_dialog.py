import configuration
import utils
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

        self.local_models_path = DirectoryPathWidget('Local Models Path')
        self.local_models_path.set_path(self.settings.value('local_models_path'))

        self.reduce_memory = QCheckBox('Reduce Memory')
        self.reduce_memory.setChecked(self.settings.value('reduce_memory', type=bool))

        self.safety_checker = QCheckBox('Safety Checker')
        self.safety_checker.setChecked(self.settings.value('safety_checker', type=bool))

        models_group = QGroupBox()

        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Repository ID"])
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
        layout.addWidget(self.local_models_path)
        layout.addWidget(self.reduce_memory)
        layout.addWidget(self.safety_checker)
        layout.addWidget(models_group)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setMinimumWidth(600)

        self.load_models()

    def load_models(self):
        huggingface_models = utils.deserialize_string_list(self.settings.value('huggingface_models'))

        self.table.setRowCount(len(huggingface_models))

        for i, repo_id in enumerate(huggingface_models):
            self.table.setItem(i, 0, QTableWidgetItem(repo_id))

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
            repo_id = repo_id_edit.text()

            if not repo_id:
                return

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(repo_id))

    def accept(self):
        huggingface_models = []
        for row in range(self.table.rowCount()):
            repo_id = self.table.item(row, 0).text()
            huggingface_models.append(repo_id)

        self.settings.setValue('local_models_path', self.local_models_path.path())
        self.settings.setValue('reduce_memory', self.reduce_memory.isChecked())
        self.settings.setValue('safety_checker', self.safety_checker.isChecked())
        self.settings.setValue('huggingface_models', huggingface_models)

        configuration.load_from_settings(self.settings)

        super().accept()
