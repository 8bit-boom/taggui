import re

from PySide6.QtWidgets import (QDialog, QFormLayout, QLineEdit,
                               QPlainTextEdit, QPushButton, QVBoxLayout)


def split_tags(tags_text: str) -> list[str]:
    tags = re.split(r'(?<!\\)[,\n]', tags_text)
    tags = [tag.strip().replace(r'\,', ',') for tag in tags]
    return [tag for tag in tags if tag]


class EditTagPresetDialog(QDialog):
    def __init__(self, parent, name: str = '', tags: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle('Edit Tag Preset' if name else 'New Tag Preset')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        form_layout = QFormLayout()
        self.name_line_edit = QLineEdit(name)
        form_layout.addRow('Name', self.name_line_edit)
        self.tags_text_edit = QPlainTextEdit(
            ', '.join(tags) if tags else '')
        self.tags_text_edit.setPlaceholderText(
            'Comma- or newline-separated tags')
        form_layout.addRow('Tags', self.tags_text_edit)
        layout.addLayout(form_layout)
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.accept)
        layout.addWidget(self.save_button)

    def get_name(self) -> str:
        return self.name_line_edit.text().strip()

    def get_tags(self) -> list[str]:
        return split_tags(self.tags_text_edit.toPlainText())
