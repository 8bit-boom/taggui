import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QMessageBox, QPushButton, QVBoxLayout)

from auto_captioning.model_manager_thread import ModelManagerThread
from auto_captioning.models_list import (MODELS, get_user_models,
                                         save_user_models)
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.utils import get_confirmation_dialog_reply

# Config keys whose presence suggests that a model accepts image inputs.
VISION_CONFIG_KEYS = ('vision_config', 'vision_model_type', 'image_token_id',
                      'image_token_index', 'mm_vision_tower',
                      'vision_feature_layer', 'vision_tower')


class ModelManagerDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.settings = get_settings()
        self.download_thread = None
        self.setWindowTitle('Model Manager')
        self.setMinimumSize(500, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(QLabel('Your models'))
        self.user_models_list = QListWidget()
        layout.addWidget(self.user_models_list)
        delete_button_layout = QHBoxLayout()
        delete_button_layout.addStretch()
        self.delete_button = QPushButton('Delete Selected')
        self.delete_button.clicked.connect(self.delete_selected_model)
        delete_button_layout.addWidget(self.delete_button)
        layout.addLayout(delete_button_layout)

        add_from_hub_group = QGroupBox('Add from Hugging Face')
        add_from_hub_layout = QVBoxLayout(add_from_hub_group)
        repo_id_layout = QHBoxLayout()
        self.repo_id_line_edit = QLineEdit()
        self.repo_id_line_edit.setPlaceholderText(
            'Repository ID, e.g. Qwen/Qwen3-VL-8B-Instruct')
        self.repo_id_line_edit.textChanged.connect(self.reset_check_result)
        self.check_button = QPushButton('Check')
        self.check_button.clicked.connect(self.check_repo)
        repo_id_layout.addWidget(self.repo_id_line_edit)
        repo_id_layout.addWidget(self.check_button)
        add_from_hub_layout.addLayout(repo_id_layout)
        self.check_result_label = QLabel()
        self.check_result_label.setWordWrap(True)
        add_from_hub_layout.addWidget(self.check_result_label)
        download_buttons_layout = QHBoxLayout()
        self.download_button = QPushButton('Download and Add')
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_and_add_model)
        self.add_without_downloading_button = QPushButton(
            'Add Without Downloading')
        self.add_without_downloading_button.setEnabled(False)
        self.add_without_downloading_button.clicked.connect(
            self.add_model_without_downloading)
        download_buttons_layout.addWidget(self.download_button)
        download_buttons_layout.addWidget(self.add_without_downloading_button)
        add_from_hub_layout.addLayout(download_buttons_layout)
        layout.addWidget(add_from_hub_group)

        add_local_group = QGroupBox('Add a Local Model Folder')
        add_local_layout = QHBoxLayout(add_local_group)
        add_local_folder_button = QPushButton('Add Local Model Folder...')
        add_local_folder_button.clicked.connect(self.add_local_model_folder)
        add_local_layout.addWidget(add_local_folder_button)
        layout.addWidget(add_local_group)

        layout.addWidget(QLabel('Built-in models (for reference)'))
        self.built_in_models_list = QListWidget()
        self.built_in_models_list.addItems(MODELS)
        self.built_in_models_list.setEnabled(False)
        layout.addWidget(self.built_in_models_list)

        self.reload_user_models_list()

    def reload_user_models_list(self):
        self.user_models_list.clear()
        self.user_models_list.addItems(get_user_models())

    def get_models_directory_path(self) -> Path | None:
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        return Path(models_directory_path) if models_directory_path else None

    def add_user_model(self, model_id: str):
        user_models = get_user_models()
        if model_id in user_models or model_id in MODELS:
            QMessageBox.information(
                self, 'Model Manager',
                f'"{model_id}" is already in the models list.')
            return
        user_models.append(model_id)
        save_user_models(user_models)
        self.reload_user_models_list()

    @Slot()
    def delete_selected_model(self):
        selected_items = self.user_models_list.selectedItems()
        if not selected_items:
            return
        model_id = selected_items[0].text()
        models_directory_path = self.get_models_directory_path()
        local_model_path = (models_directory_path / model_id
                            if models_directory_path else Path(model_id))
        delete_files = False
        if local_model_path.is_dir():
            reply = QMessageBox.question(
                self, 'Delete Model',
                f'Also delete the downloaded files for "{model_id}" from '
                f'disk?',
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            delete_files = reply == QMessageBox.StandardButton.Yes
        else:
            reply = get_confirmation_dialog_reply(
                title='Delete Model',
                question=f'Remove "{model_id}" from your models list?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        user_models = get_user_models()
        if model_id in user_models:
            user_models.remove(model_id)
            save_user_models(user_models)
        if delete_files:
            shutil.rmtree(local_model_path, ignore_errors=True)
        self.reload_user_models_list()

    @Slot()
    def reset_check_result(self):
        self.check_result_label.clear()
        self.download_button.setEnabled(False)
        self.add_without_downloading_button.setEnabled(False)

    @Slot()
    def check_repo(self):
        repo_id = self.repo_id_line_edit.text().strip()
        self.download_button.setEnabled(False)
        self.add_without_downloading_button.setEnabled(False)
        if not repo_id:
            self.check_result_label.setText('Enter a repository ID.')
            return
        self.check_result_label.setText('Checking...')
        QApplication.processEvents()
        try:
            is_wd_tagger_model = False
            try:
                hf_hub_download(repo_id, filename='selected_tags.csv')
                is_wd_tagger_model = True
            except (HfHubHTTPError, OSError):
                pass
            if is_wd_tagger_model:
                self.check_result_label.setText(
                    f'"{repo_id}" looks like a WD Tagger model.')
            else:
                try:
                    config_path = hf_hub_download(repo_id,
                                                   filename='config.json')
                except (HfHubHTTPError, OSError) as exception:
                    self.check_result_label.setText(
                        f'Could not find a config.json for "{repo_id}": '
                        f'{exception}')
                    return
                config_text = Path(config_path).read_text(
                    encoding='utf-8', errors='replace').lower()
                if any(key in config_text for key in VISION_CONFIG_KEYS):
                    self.check_result_label.setText(
                        f'"{repo_id}" looks like a vision/captioning model.')
                else:
                    self.check_result_label.setText(
                        f'"{repo_id}" does not look like a vision/'
                        f'captioning model. It may not work correctly for '
                        f'auto-captioning.')
        except Exception as exception:
            self.check_result_label.setText(
                f'Failed to check "{repo_id}": {exception}')
            return
        self.download_button.setEnabled(True)
        self.add_without_downloading_button.setEnabled(True)

    @Slot()
    def add_model_without_downloading(self):
        repo_id = self.repo_id_line_edit.text().strip()
        if repo_id:
            self.add_user_model(repo_id)

    @Slot()
    def download_and_add_model(self):
        repo_id = self.repo_id_line_edit.text().strip()
        if not repo_id:
            return
        models_directory_path = self.get_models_directory_path()
        if not models_directory_path:
            QMessageBox.warning(
                self, 'Model Manager',
                'Set an auto-captioning models directory in Settings before '
                'downloading models.')
            return
        local_directory_path = models_directory_path / repo_id
        self.download_button.setEnabled(False)
        self.add_without_downloading_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.check_result_label.setText(f'Downloading {repo_id}...')
        self.download_thread = ModelManagerThread(self, repo_id,
                                                   local_directory_path)
        self.download_thread.download_finished.connect(
            self.handle_download_finished)
        self.download_thread.start()

    @Slot(bool, str)
    def handle_download_finished(self, success: bool, error_message: str):
        repo_id = self.repo_id_line_edit.text().strip()
        self.download_button.setEnabled(True)
        self.add_without_downloading_button.setEnabled(True)
        self.check_button.setEnabled(True)
        if not success:
            self.check_result_label.setText(
                f'Failed to download "{repo_id}": {error_message}')
            QMessageBox.critical(self, 'Model Manager',
                                 f'Failed to download "{repo_id}": '
                                 f'{error_message}')
            return
        self.check_result_label.setText(f'Downloaded "{repo_id}".')
        self.add_user_model(repo_id)

    @Slot()
    def add_local_model_folder(self):
        initial_directory = self.settings.value('directory_path',
                                                 type=str) or ''
        folder_path = QFileDialog.getExistingDirectory(
            parent=self, caption='Select local model folder',
            dir=initial_directory)
        if not folder_path:
            return
        folder_path = Path(folder_path)
        config_path = folder_path / 'config.json'
        tags_path = folder_path / 'selected_tags.csv'
        if not config_path.is_file() and not tags_path.is_file():
            reply = get_confirmation_dialog_reply(
                title='Add Local Model Folder',
                question=f'"{folder_path}" does not look like a model '
                         f'folder (no config.json or selected_tags.csv was '
                         f'found there). Add it anyway?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.add_user_model(str(folder_path))
