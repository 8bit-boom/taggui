import statistics
from collections import Counter
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QDockWidget,
                               QFormLayout, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)
from transformers import PreTrainedTokenizerBase

from models.image_list_model import ImageListModel
from models.tag_counter_model import TagCounterModel
from utils.big_widgets import TallPushButton
from utils.image import Image
from utils.settings_widgets import (FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox)

# Matches `ImageTagsEditor.MAX_TOKEN_COUNT`.
MAX_TOKEN_COUNT = 75
ASPECT_RATIO_TOLERANCE = 0.08
ASPECT_RATIO_BUCKETS = {
    '1:1': 1 / 1,
    '4:3': 4 / 3,
    '3:4': 3 / 4,
    '16:9': 16 / 9,
    '9:16': 9 / 16,
}
AUTO_REFRESH_DEBOUNCE_MS = 750


def get_aspect_ratio_bucket(width: int, height: int) -> str:
    if not height:
        return 'other'
    ratio = width / height
    for bucket_name, bucket_ratio in ASPECT_RATIO_BUCKETS.items():
        if abs(ratio - bucket_ratio) < ASPECT_RATIO_TOLERANCE:
            return bucket_name
    return 'other'


def get_stats(values: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, median, min, max), or all zeros for an empty list."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    return (statistics.mean(values), statistics.median(values), min(values),
            max(values))


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Raised)


@dataclass
class Statistics:
    image_count: int = 0
    tagged_image_count: int = 0
    untagged_image_count: int = 0
    missing_caption_file_count: int = 0
    tag_instance_count: int = 0
    unique_tag_count: int = 0
    singleton_tag_count: int = 0
    tags_per_image: tuple[float, float, float, float] = (0, 0, 0, 0)
    caption_chars: tuple[float, float, float, float] = (0, 0, 0, 0)
    token_counts: tuple[float, float, float, float] = (0, 0, 0, 0)
    over_token_limit_count: int = 0
    duplicate_caption_groups: int = 0
    aspect_ratio_buckets: Counter = field(default_factory=Counter)
    megapixels: tuple[float, float, float, float] = (0, 0, 0, 0)
    small_image_count: int = 0
    format_counts: Counter = field(default_factory=Counter)
    total_bytes: int = 0


def compute_statistics(images: list[Image], tag_counter: Counter,
                       tokenizer: PreTrainedTokenizerBase, tag_separator: str,
                       min_side: int) -> Statistics:
    statistics_ = Statistics()
    statistics_.image_count = len(images)
    statistics_.tagged_image_count = sum(1 for image in images if image.tags)
    statistics_.untagged_image_count = (statistics_.image_count
                                        - statistics_.tagged_image_count)
    statistics_.missing_caption_file_count = sum(
        1 for image in images if not image.has_caption_file)
    statistics_.tag_instance_count = sum(tag_counter.values())
    statistics_.unique_tag_count = len(tag_counter)
    statistics_.singleton_tag_count = sum(1 for count in tag_counter.values()
                                          if count == 1)
    statistics_.tags_per_image = get_stats(
        [len(image.tags) for image in images])
    captions = [tag_separator.join(image.tags) for image in images]
    statistics_.caption_chars = get_stats([len(caption)
                                          for caption in captions])
    if images:
        # Subtract 2 for the `<|startoftext|>` and `<|endoftext|>` tokens.
        token_counts = [len(input_ids) - 2 for input_ids
                        in tokenizer(captions).input_ids]
    else:
        token_counts = []
    statistics_.token_counts = get_stats(token_counts)
    statistics_.over_token_limit_count = sum(
        1 for token_count in token_counts if token_count > MAX_TOKEN_COUNT)
    caption_counter = Counter(caption for caption in captions if caption)
    statistics_.duplicate_caption_groups = sum(
        1 for count in caption_counter.values() if count > 1)
    megapixels = []
    for image in images:
        if image.dimensions is None:
            continue
        width, height = image.dimensions
        statistics_.aspect_ratio_buckets[
            get_aspect_ratio_bucket(width, height)] += 1
        megapixels.append(width * height / 1_000_000)
        if min(width, height) < min_side:
            statistics_.small_image_count += 1
    statistics_.megapixels = get_stats(megapixels)
    statistics_.format_counts = Counter(
        image.path.suffix.lstrip('.').lower() for image in images)
    statistics_.total_bytes = sum(image.file_size for image in images
                                  if image.file_size)
    return statistics_


class DatasetStatisticsPane(QDockWidget):
    image_list_filter_requested = Signal(str)

    def __init__(self, image_list_model: ImageListModel,
                 tag_counter_model: TagCounterModel,
                 tokenizer: PreTrainedTokenizerBase, tag_separator: str):
        super().__init__()
        self.image_list_model = image_list_model
        self.tag_counter_model = tag_counter_model
        self.tokenizer = tokenizer
        self.tag_separator = tag_separator
        self.statistics = Statistics()

        self.setObjectName('dataset_statistics')
        self.setWindowTitle('Statistics')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.refresh_button = TallPushButton('Refresh')
        self.refresh_button.clicked.connect(self.refresh)
        self.auto_refresh_check_box = SettingsBigCheckBox(
            key='statistics_auto_refresh', default=False)
        auto_refresh_layout = QHBoxLayout()
        auto_refresh_layout.addWidget(QLabel('Auto-refresh'))
        auto_refresh_layout.addWidget(self.auto_refresh_check_box)
        self.min_side_spin_box = FocusedScrollSettingsSpinBox(
            key='statistics_min_side', default=512, minimum=1, maximum=99999)
        min_side_layout = QHBoxLayout()
        min_side_layout.addWidget(QLabel('Small image threshold (px)'))
        min_side_layout.addWidget(self.min_side_spin_box, stretch=1)
        self.min_side_spin_box.valueChanged.connect(
            lambda _: self.refresh())
        self.copy_report_button = QPushButton('Copy Report')
        self.copy_report_button.clicked.connect(self.copy_report)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_labels: dict[str, QLabel] = {}
        self.add_row(form_layout, 'Images', 'image_count')
        self.add_row(form_layout, 'Tagged', 'tagged_image_count',
                     filter_text='tags:>0')
        self.add_row(form_layout, 'Untagged', 'untagged_image_count',
                     filter_text='tags:=0')
        self.add_row(form_layout, 'Missing caption file',
                     'missing_caption_file_count')
        form_layout.addRow(HorizontalLine())
        self.add_row(form_layout, 'Unique tags', 'unique_tag_count')
        self.add_row(form_layout, 'Tag instances', 'tag_instance_count')
        self.add_row(form_layout, 'Tags used once', 'singleton_tag_count')
        self.add_row(form_layout, 'Tags per image (mean/median/min/max)',
                     'tags_per_image')
        form_layout.addRow(HorizontalLine())
        self.add_row(form_layout, 'Caption chars (mean/median/min/max)',
                     'caption_chars')
        self.add_row(form_layout, 'Tokens (mean/median/min/max)',
                     'token_counts')
        self.add_row(form_layout, f'Over {MAX_TOKEN_COUNT} tokens',
                     'over_token_limit_count', filter_text='tokens:>75')
        self.add_row(form_layout, 'Duplicate caption groups',
                     'duplicate_caption_groups')
        form_layout.addRow(HorizontalLine())
        self.add_row(form_layout, 'Megapixels (mean/median/min/max)',
                     'megapixels')
        self.add_row(form_layout, 'Small images', 'small_image_count',
                     filter_text=None)
        self.add_row(form_layout, 'Total size on disk', 'total_bytes')
        self.aspect_ratio_label = QLabel()
        self.aspect_ratio_label.setWordWrap(True)
        form_layout.addRow('Aspect ratios', self.aspect_ratio_label)
        self.format_label = QLabel()
        self.format_label.setWordWrap(True)
        form_layout.addRow('Formats', self.format_label)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(self.refresh_button)
        container_layout.addLayout(auto_refresh_layout)
        container_layout.addLayout(min_side_layout)
        container_layout.addLayout(form_layout)
        container_layout.addWidget(self.copy_report_button)
        container_layout.addStretch()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(AUTO_REFRESH_DEBOUNCE_MS)
        self.refresh_timer.timeout.connect(self.refresh)

    def add_row(self, form_layout: QFormLayout, label_text: str,
               statistics_field: str, filter_text: str | None = ''):
        value_label = QLabel()
        self.value_labels[statistics_field] = value_label
        if filter_text:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(value_label, stretch=1)
            show_button = QPushButton('Show')
            show_button.clicked.connect(
                lambda: self.image_list_filter_requested.emit(filter_text))
            row_layout.addWidget(show_button)
            form_layout.addRow(label_text, row_widget)
        else:
            form_layout.addRow(label_text, value_label)

    @Slot()
    def request_refresh_if_auto_refresh_enabled(self):
        if (self.auto_refresh_check_box.isChecked() and self.isVisible()):
            self.refresh_timer.start()

    @staticmethod
    def format_stats_tuple(values: tuple[float, float, float, float],
                           decimals: int = 1) -> str:
        mean, median, minimum, maximum = values
        return (f'{mean:.{decimals}f} / {median:.{decimals}f} / '
                f'{minimum:.{decimals}f} / {maximum:.{decimals}f}')

    @staticmethod
    def format_bytes(total_bytes: int) -> str:
        size = float(total_bytes)
        for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
            if size < 1024 or unit == 'TB':
                return f'{size:.1f} {unit}'
            size /= 1024
        return f'{size:.1f} TB'

    @Slot()
    def refresh(self):
        self.statistics = compute_statistics(
            self.image_list_model.images, self.tag_counter_model.tag_counter,
            self.tokenizer, self.tag_separator,
            self.min_side_spin_box.value())
        simple_fields = (
            'image_count', 'tagged_image_count', 'untagged_image_count',
            'missing_caption_file_count', 'unique_tag_count',
            'tag_instance_count', 'singleton_tag_count',
            'over_token_limit_count', 'duplicate_caption_groups',
            'small_image_count')
        for field_name in simple_fields:
            self.value_labels[field_name].setText(
                str(getattr(self.statistics, field_name)))
        for field_name in ('tags_per_image', 'caption_chars', 'token_counts',
                          'megapixels'):
            self.value_labels[field_name].setText(
                self.format_stats_tuple(getattr(self.statistics, field_name)))
        self.value_labels['total_bytes'].setText(
            self.format_bytes(self.statistics.total_bytes))
        aspect_ratio_text = ', '.join(
            f'{bucket}: {count}' for bucket, count
            in self.statistics.aspect_ratio_buckets.most_common())
        self.aspect_ratio_label.setText(aspect_ratio_text or '(none)')
        format_text = ', '.join(
            f'{format_}: {count}' for format_, count
            in self.statistics.format_counts.most_common())
        self.format_label.setText(format_text or '(none)')

    @Slot()
    def copy_report(self):
        statistics_ = self.statistics
        lines = [
            f'Images: {statistics_.image_count} '
            f'({statistics_.tagged_image_count} tagged, '
            f'{statistics_.untagged_image_count} untagged, '
            f'{statistics_.missing_caption_file_count} missing caption '
            f'file)',
            f'Unique tags: {statistics_.unique_tag_count} '
            f'({statistics_.tag_instance_count} instances, '
            f'{statistics_.singleton_tag_count} used once)',
            'Tags per image (mean/median/min/max): '
            f'{self.format_stats_tuple(statistics_.tags_per_image)}',
            'Caption chars (mean/median/min/max): '
            f'{self.format_stats_tuple(statistics_.caption_chars)}',
            'Tokens (mean/median/min/max): '
            f'{self.format_stats_tuple(statistics_.token_counts)} '
            f'({statistics_.over_token_limit_count} over '
            f'{MAX_TOKEN_COUNT})',
            f'Duplicate caption groups: '
            f'{statistics_.duplicate_caption_groups}',
            'Megapixels (mean/median/min/max): '
            f'{self.format_stats_tuple(statistics_.megapixels)}',
            f'Small images (< {self.min_side_spin_box.value()}px): '
            f'{statistics_.small_image_count}',
            f'Total size on disk: '
            f'{self.format_bytes(statistics_.total_bytes)}',
            'Aspect ratios: ' + ', '.join(
                f'{bucket}: {count}' for bucket, count
                in statistics_.aspect_ratio_buckets.most_common()),
            'Formats: ' + ', '.join(
                f'{format_}: {count}' for format_, count
                in statistics_.format_counts.most_common()),
        ]
        QApplication.clipboard().setText('\n'.join(lines))
