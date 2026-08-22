import csv
import json
from enum import Enum
from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout,
                               QLabel, QMessageBox, QVBoxLayout)
from transformers import PreTrainedTokenizerBase

from models.image_list_model import ImageListModel, Scope
from models.tag_counter_model import TagCounterModel
from utils.big_widgets import TallPushButton
from utils.settings_widgets import SettingsComboBox


class ExportFormat(str, Enum):
    CSV = 'CSV'
    JSON = 'JSON'
    JSONL = 'JSONL (metadata.jsonl)'
    TAG_FREQUENCY_CSV = 'Tag frequency CSV'
    FILE_LIST = 'File list (.txt)'


FORMAT_EXTENSIONS = {
    ExportFormat.CSV: '.csv',
    ExportFormat.JSON: '.json',
    ExportFormat.JSONL: '.jsonl',
    ExportFormat.TAG_FREQUENCY_CSV: '.csv',
    ExportFormat.FILE_LIST: '.txt',
}


class ExportDatasetDialog(QDialog):
    def __init__(self, parent, image_list_model: ImageListModel,
                 tag_counter_model: TagCounterModel,
                 tokenizer: PreTrainedTokenizerBase, tag_separator: str):
        super().__init__(parent)
        self.image_list_model = image_list_model
        self.tag_counter_model = tag_counter_model
        self.tokenizer = tokenizer
        self.tag_separator = tag_separator
        self.setWindowTitle('Export Dataset')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        description_label = QLabel(
            'Export image paths, captions, and tags to a file usable '
            'outside TagGUI (for training scripts, backups, or sharing '
            'tag frequency data).')
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        form_layout = QFormLayout()
        self.scope_combo_box = SettingsComboBox(key='export_scope')
        self.scope_combo_box.addItems(list(Scope))
        form_layout.addRow('Scope', self.scope_combo_box)
        self.format_combo_box = QComboBox()
        self.format_combo_box.addItems(list(ExportFormat))
        form_layout.addRow('Format', self.format_combo_box)
        layout.addLayout(form_layout)

        self.export_button = TallPushButton('Export...')
        self.export_button.clicked.connect(self.export)
        layout.addWidget(self.export_button)

    def get_images_in_scope(self) -> list:
        scope = self.scope_combo_box.currentText()
        return [image for image_index, image
                in enumerate(self.image_list_model.images)
                if self.image_list_model.is_image_in_scope(
                    scope, image_index, image)]

    @Slot()
    def export(self):
        export_format = self.format_combo_box.currentText()
        extension = FORMAT_EXTENSIONS[export_format]
        destination_path_string, _ = QFileDialog.getSaveFileName(
            parent=self, caption='Export dataset to',
            filter=f'*{extension}')
        if not destination_path_string:
            return
        destination_path = Path(destination_path_string)
        if destination_path.suffix.lower() != extension:
            destination_path = destination_path.with_suffix(extension)
        images = self.get_images_in_scope()
        try:
            if export_format == ExportFormat.CSV:
                self.export_csv(destination_path, images)
            elif export_format == ExportFormat.JSON:
                self.export_json(destination_path, images)
            elif export_format == ExportFormat.JSONL:
                self.export_jsonl(destination_path, images)
            elif export_format == ExportFormat.TAG_FREQUENCY_CSV:
                self.export_tag_frequency_csv(destination_path)
            elif export_format == ExportFormat.FILE_LIST:
                self.export_file_list(destination_path, images)
        except OSError as exception:
            QMessageBox.critical(self, 'Export Dataset',
                                 f'Failed to export to {destination_path}: '
                                 f'{exception}')
            return
        message_box = QMessageBox(self)
        message_box.setWindowTitle('Export Dataset')
        message_box.setIcon(QMessageBox.Icon.Information)
        message_box.setText(f'Exported {len(images)} images to '
                            f'{destination_path}.')
        message_box.exec()

    def get_token_count(self, caption: str) -> int:
        # Subtract 2 for the `<|startoftext|>` and `<|endoftext|>` tokens.
        return len(self.tokenizer(caption).input_ids) - 2

    def export_csv(self, destination_path: Path, images: list):
        with open(destination_path, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['path', 'caption', 'tags', 'width', 'height',
                             'chars', 'tokens'])
            for image in images:
                caption = self.tag_separator.join(image.tags)
                width, height = image.dimensions or (None, None)
                writer.writerow([str(image.path), caption,
                                self.tag_separator.join(image.tags), width,
                                height, len(caption),
                                self.get_token_count(caption)])

    def export_json(self, destination_path: Path, images: list):
        records = [
            {
                'path': str(image.path),
                'caption': self.tag_separator.join(image.tags),
                'tags': image.tags,
            }
            for image in images
        ]
        with open(destination_path, 'w', encoding='utf-8') as json_file:
            json.dump(records, json_file, indent=2, ensure_ascii=False)

    def export_jsonl(self, destination_path: Path, images: list):
        export_root = destination_path.parent
        with open(destination_path, 'w', encoding='utf-8') as jsonl_file:
            for image in images:
                try:
                    file_name = str(image.path.relative_to(export_root))
                except ValueError:
                    file_name = image.path.name
                record = {'file_name': file_name,
                         'text': self.tag_separator.join(image.tags)}
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def export_tag_frequency_csv(self, destination_path: Path):
        with open(destination_path, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['name', 'category', 'post_count', 'aliases'])
            for tag, post_count in (self.tag_counter_model.tag_counter
                                    .most_common()):
                writer.writerow([tag, '', post_count, ''])

    def export_file_list(self, destination_path: Path, images: list):
        with open(destination_path, 'w', encoding='utf-8') as text_file:
            for image in images:
                text_file.write(f'{image.path}\n')
