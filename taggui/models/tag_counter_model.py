import csv
from collections import Counter
from pathlib import Path

from PySide6.QtCore import QAbstractListModel, Qt, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from utils.image import Image
from utils.utils import get_confirmation_dialog_reply, list_with_and, pluralize


class TagCounterModel(QAbstractListModel):
    tags_renaming_requested = Signal(list, str)

    def __init__(self):
        super().__init__()
        self.tag_counter = Counter()
        self.most_common_tags = []
        self.all_tags_list = None

    def rowCount(self, parent=None) -> int:
        return len(self.most_common_tags)

    def data(self, index, role=None) -> tuple[str, int] | str:
        tag, count = self.most_common_tags[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return tag, count
        if role == Qt.ItemDataRole.DisplayRole:
            return f'{tag} ({count})'
        if role == Qt.ItemDataRole.EditRole:
            return tag

    def flags(self, index) -> Qt.ItemFlag:
        """Make the tags editable."""
        return (Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsEnabled)

    def setData(self, index, value: str,
                role=Qt.ItemDataRole.EditRole) -> bool:
        new_tag = value
        if not new_tag or role != Qt.ItemDataRole.EditRole:
            return False
        old_tag = self.data(index, Qt.ItemDataRole.EditRole)
        if new_tag == old_tag:
            return False
        selected_indices = self.all_tags_list.selectedIndexes()
        old_tags = []
        old_tags_count = 0
        for selected_index in selected_indices:
            old_tag, old_tag_count = selected_index.data(
                Qt.ItemDataRole.UserRole)
            old_tags.append(old_tag)
            old_tags_count += old_tag_count
        question = (f'Rename {old_tags_count} '
                    f'{pluralize("instance", old_tags_count)} of ')
        if len(old_tags) < 10:
            quoted_tags = [f'"{tag}"' for tag in old_tags]
            question += (f'{pluralize("tag", len(old_tags))} '
                         f'{list_with_and(quoted_tags)} ')
        else:
            question += f'{len(old_tags)} tags '
        question += f'to "{new_tag}"?'
        reply = get_confirmation_dialog_reply(
            title=f'Rename {pluralize("Tag", len(old_tags))}',
            question=question)
        if reply == QMessageBox.StandardButton.Yes:
            self.tags_renaming_requested.emit(old_tags, new_tag)
            return True
        return False

    @Slot()
    def count_tags(self, images: list[Image]):
        self.tag_counter.clear()
        for image in images:
            self.tag_counter.update(image.tags)
        self.most_common_tags = self.tag_counter.most_common()
        self.modelReset.emit()


def load_danbooru_tags_csv(csv_path: str | Path) -> list[str]:
    """
    Load tags from a Danbooru-format tag CSV file (with columns `name`,
    `category`, `post_count`, `aliases`), sorted by post count in descending
    order.
    """
    tags_and_post_counts = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames and 'name' in reader.fieldnames:
            for row in reader:
                try:
                    post_count = int(row.get('post_count') or 0)
                except ValueError:
                    post_count = 0
                tags_and_post_counts.append((row['name'], post_count))
        else:
            # No recognized header; assume the columns are in the order
            # `name, category, post_count, aliases`.
            csv_file.seek(0)
            for row in csv.reader(csv_file):
                if len(row) < 3:
                    continue
                try:
                    post_count = int(row[2])
                except ValueError:
                    continue
                tags_and_post_counts.append((row[0], post_count))
    tags_and_post_counts.sort(key=lambda tag_and_post_count: tag_and_post_count[1],
                              reverse=True)
    return [tag for tag, _ in tags_and_post_counts]


class DanbooruTagCompletionModel(QAbstractListModel):
    """
    A tag autocomplete source that merges the tags already used in the
    dataset (shown first, from `TagCounterModel`) with tags loaded from an
    external Danbooru-format tag CSV file, sorted by post count. Danbooru
    tags that are already in the dataset are excluded so that each tag only
    appears once.
    """

    def __init__(self, tag_counter_model: TagCounterModel):
        super().__init__()
        self.tag_counter_model = tag_counter_model
        self.danbooru_tags: list[str] = []
        self.combined_tags: list[str] = []
        self.tag_counter_model.modelReset.connect(self.update_combined_tags)

    def load_danbooru_tags_csv(self, csv_path: str | Path):
        try:
            self.danbooru_tags = load_danbooru_tags_csv(csv_path)
        except OSError as exception:
            print(f'Failed to load the Danbooru tag autocomplete CSV at '
                  f'{csv_path}: {exception}')
            self.danbooru_tags = []
        self.update_combined_tags()

    @Slot()
    def update_combined_tags(self):
        self.beginResetModel()
        dataset_tags = [tag for tag, _
                        in self.tag_counter_model.most_common_tags]
        dataset_tags_set = set(dataset_tags)
        danbooru_tags = [tag for tag in self.danbooru_tags
                         if tag not in dataset_tags_set]
        self.combined_tags = dataset_tags + danbooru_tags
        self.endResetModel()

    def rowCount(self, parent=None) -> int:
        return len(self.combined_tags)

    def data(self, index, role=None) -> str | None:
        tag = self.combined_tags[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return tag
        return None
