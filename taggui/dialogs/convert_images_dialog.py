from PIL import Image as PilImage
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout,
                               QLabel, QMessageBox, QSlider, QVBoxLayout,
                               QWidget)

from utils.big_widgets import TallPushButton
from utils.image import Image
from utils.settings_widgets import SettingsBigCheckBox
from utils.utils import pluralize

LOSSY_FORMATS = ('JPEG', 'WEBP', 'AVIF')
FORMAT_EXTENSIONS = {
    'PNG': '.png',
    'JPEG': '.jpg',
    'WEBP': '.webp',
    'AVIF': '.avif',
    'BMP': '.bmp',
    'TIFF': '.tiff',
    'GIF': '.gif'
}


class ConvertImagesDialog(QDialog):
    images_converted = Signal()

    def __init__(self, parent, images: list[Image]):
        super().__init__(parent)
        self.images = images
        self.setWindowTitle('Convert Images')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        image_count = len(images)
        description_label = QLabel(
            f'Convert {image_count} selected '
            f'{pluralize("image", image_count)} to another format. The '
            f'converted files keep their original names, so existing '
            f'captions will still match them.')
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        form_layout = QFormLayout()
        self.format_combo_box = QComboBox()
        self.format_combo_box.addItems(list(FORMAT_EXTENSIONS))
        self.format_combo_box.currentTextChanged.connect(
            self.set_quality_row_visibility)
        form_layout.addRow('Convert to', self.format_combo_box)
        self.quality_row_widget = QWidget()
        quality_row_layout = QHBoxLayout(self.quality_row_widget)
        quality_row_layout.setContentsMargins(0, 0, 0, 0)
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(90)
        self.quality_value_label = QLabel('90')
        self.quality_value_label.setMinimumWidth(30)
        self.quality_slider.valueChanged.connect(
            lambda value: self.quality_value_label.setText(str(value)))
        quality_row_layout.addWidget(self.quality_slider)
        quality_row_layout.addWidget(self.quality_value_label)
        self.quality_row_label = QLabel('Quality')
        form_layout.addRow(self.quality_row_label, self.quality_row_widget)
        layout.addLayout(form_layout)

        delete_originals_layout = QHBoxLayout()
        self.delete_originals_check_box = SettingsBigCheckBox(
            key='convert_images_delete_originals', default=False)
        delete_originals_layout.addWidget(QLabel('Delete original files'))
        delete_originals_layout.addWidget(self.delete_originals_check_box)
        delete_originals_layout.addStretch()
        layout.addLayout(delete_originals_layout)

        self.convert_button = TallPushButton(
            f'Convert {pluralize("Image", image_count)}')
        self.convert_button.clicked.connect(self.convert_images)
        layout.addWidget(self.convert_button)

        self.set_quality_row_visibility(self.format_combo_box.currentText())

    @Slot(str)
    def set_quality_row_visibility(self, format_: str):
        is_lossy_format = format_ in LOSSY_FORMATS
        self.quality_row_label.setVisible(is_lossy_format)
        self.quality_row_widget.setVisible(is_lossy_format)

    @Slot()
    def convert_images(self):
        format_ = self.format_combo_box.currentText()
        extension = FORMAT_EXTENSIONS[format_]
        quality = self.quality_slider.value()
        delete_originals = self.delete_originals_check_box.isChecked()
        converted_count = 0
        skipped_count = 0
        failed_count = 0
        for image in self.images:
            output_path = image.path.with_suffix(extension)
            if output_path == image.path or output_path.exists():
                skipped_count += 1
                continue
            try:
                with PilImage.open(image.path) as pil_image:
                    if format_ in ('JPEG', 'BMP'):
                        pil_image = pil_image.convert('RGB')
                    save_arguments = {}
                    if format_ in LOSSY_FORMATS:
                        save_arguments['quality'] = quality
                    pil_image.save(output_path, format=format_,
                                   **save_arguments)
            except (OSError, ValueError) as exception:
                failed_count += 1
                print(f'Failed to convert {image.path} to {format_}: '
                      f'{exception}')
                continue
            converted_count += 1
            if delete_originals:
                try:
                    image.path.unlink()
                except OSError as exception:
                    print(f'Failed to delete {image.path}: {exception}')
        message = (f'Converted {converted_count} '
                  f'{pluralize("image", converted_count)} to {format_}.')
        if skipped_count:
            message += (f'\nSkipped {skipped_count} '
                        f'{pluralize("image", skipped_count)} because a '
                        f'file already existed at the target path.')
        if failed_count:
            message += (f'\nFailed to convert {failed_count} '
                        f'{pluralize("image", failed_count)}.')
        message_box = QMessageBox(self)
        message_box.setWindowTitle('Convert Images')
        message_box.setIcon(QMessageBox.Icon.Information)
        message_box.setText(message)
        message_box.exec()
        self.images_converted.emit()
        self.accept()
