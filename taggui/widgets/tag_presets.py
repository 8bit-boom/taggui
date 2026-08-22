from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QDialog, QDockWidget, QHBoxLayout, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from dialogs.edit_tag_preset_dialog import EditTagPresetDialog
from models.image_list_model import ImageListModel
from utils.image import Image
from utils.settings import get_tag_presets, save_tag_presets
from utils.utils import get_confirmation_dialog_reply, pluralize
from widgets.image_list import ImageList


class TagPresetsPane(QDockWidget):
    """
    A dock of named tag groups ("presets") that can be applied to or removed
    from the selected images, either by button/double-click or by the
    `Alt+1`...`Alt+9` shortcuts wired up in `MainWindow`.
    """
    tags_addition_requested = Signal(list, list)
    tags_removal_requested = Signal(list, list)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.setObjectName('tag_presets')
        self.setWindowTitle('Tag Presets')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.preset_list_widget = QListWidget()
        self.preset_list_widget.itemDoubleClicked.connect(
            lambda _: self.apply_selected_preset())
        self.new_button = QPushButton('New...')
        self.new_button.clicked.connect(self.new_preset)
        self.edit_button = QPushButton('Edit...')
        self.edit_button.clicked.connect(self.edit_selected_preset)
        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(self.delete_selected_preset)
        self.apply_button = QPushButton('Apply to Selected Images')
        self.apply_button.clicked.connect(self.apply_selected_preset)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.new_button)
        buttons_layout.addWidget(self.edit_button)
        buttons_layout.addWidget(self.delete_button)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.preset_list_widget)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.apply_button)
        self.setWidget(container)

        self.reload_presets()

    def reload_presets(self):
        self.preset_list_widget.clear()
        for preset in get_tag_presets():
            self.preset_list_widget.addItem(QListWidgetItem(preset['name']))

    def get_selected_preset_index(self) -> int | None:
        selected_items = self.preset_list_widget.selectedItems()
        if not selected_items:
            return None
        return self.preset_list_widget.row(selected_items[0])

    @Slot()
    def new_preset(self):
        dialog = EditTagPresetDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = dialog.get_name()
        tags = dialog.get_tags()
        if not name or not tags:
            return
        presets = get_tag_presets()
        presets.append({'name': name, 'tags': tags})
        save_tag_presets(presets)
        self.reload_presets()

    @Slot()
    def edit_selected_preset(self):
        index = self.get_selected_preset_index()
        if index is None:
            return
        presets = get_tag_presets()
        preset = presets[index]
        dialog = EditTagPresetDialog(self, preset['name'], preset['tags'])
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = dialog.get_name()
        tags = dialog.get_tags()
        if not name or not tags:
            return
        presets[index] = {'name': name, 'tags': tags}
        save_tag_presets(presets)
        self.reload_presets()

    @Slot()
    def delete_selected_preset(self):
        index = self.get_selected_preset_index()
        if index is None:
            return
        presets = get_tag_presets()
        reply = get_confirmation_dialog_reply(
            title='Delete Tag Preset',
            question=f'Delete the tag preset "{presets[index]["name"]}"?')
        if reply != QMessageBox.StandardButton.Yes:
            return
        del presets[index]
        save_tag_presets(presets)
        self.reload_presets()

    def get_current_image_tags(self) -> list[str] | None:
        selected_image_indices = self.image_list.get_selected_image_indices()
        if not selected_image_indices:
            return None
        image: Image = self.image_list_model.data(
            selected_image_indices[0], Qt.ItemDataRole.UserRole)
        return image.tags

    def apply_preset(self, preset_index: int):
        selected_image_indices = self.image_list.get_selected_image_indices()
        presets = get_tag_presets()
        if not 0 <= preset_index < len(presets) or not selected_image_indices:
            return
        preset = presets[preset_index]
        tags = preset['tags']
        if len(selected_image_indices) > 1:
            reply = get_confirmation_dialog_reply(
                title='Apply Tag Preset',
                question=f'Add {pluralize("tag", len(tags))} from preset '
                         f'"{preset["name"]}" to '
                         f'{len(selected_image_indices)} selected images?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.tags_addition_requested.emit(tags, selected_image_indices)

    def toggle_preset(self, preset_index: int):
        """
        Remove the preset's tags if the first selected image already has all
        of them, otherwise add them to all selected images.
        """
        selected_image_indices = self.image_list.get_selected_image_indices()
        presets = get_tag_presets()
        if not 0 <= preset_index < len(presets) or not selected_image_indices:
            return
        preset = presets[preset_index]
        tags = preset['tags']
        current_image_tags = self.get_current_image_tags()
        already_applied = (current_image_tags is not None
                           and all(tag in current_image_tags
                                   for tag in tags))
        if not already_applied:
            self.apply_preset(preset_index)
            return
        if len(selected_image_indices) > 1:
            reply = get_confirmation_dialog_reply(
                title='Remove Tag Preset',
                question=f'Remove {pluralize("tag", len(tags))} from preset '
                         f'"{preset["name"]}" from '
                         f'{len(selected_image_indices)} selected images?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.tags_removal_requested.emit(tags, selected_image_indices)

    @Slot()
    def apply_selected_preset(self):
        index = self.get_selected_preset_index()
        if index is None:
            return
        self.apply_preset(index)
