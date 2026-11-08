import sys
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QInputDialog,
                               QListWidget, QMessageBox, QPlainTextEdit,
                               QPushButton, QVBoxLayout)

from auto_captioning.captioning_thread import CaptioningThread
from auto_captioning.model_chains import get_model_chains, save_model_chains
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from utils.enums import CaptionPosition
from utils.settings import DEFAULT_SETTINGS, get_settings, get_tag_separator
from utils.utils import pluralize, set_text_edit_height


def describe_step(step_settings: dict, step_number: int) -> str:
    model_id = step_settings.get('model_id', '(unknown model)')
    caption_position = step_settings.get('caption_position', '')
    prompt = step_settings.get('prompt', '').strip().replace('\n', ' ')
    if len(prompt) > 60:
        prompt = f'{prompt[:60]}...'
    description = f'{step_number}. {model_id} ({caption_position})'
    if prompt:
        description += f' - "{prompt}"'
    return description


class ModelChainDialog(QDialog):
    """
    Run a sequence of captioning passes - each with its own model and full
    settings snapshot - over the same selected images, one step after
    another. A step's model is loaded once and reused for every image
    before moving to the next step (instead of reloading per image), the
    same way running each step manually, one after another, already
    would. Built from `CaptioningThread`, the same machinery normal
    captioning uses, so each step's caption is actually applied to the
    image list (not just previewed) - typically so a later step's prompt
    can reference an earlier step's output via the `{tags}` template
    variable.
    """

    def __init__(self, auto_captioner):
        super().__init__(None)
        self.auto_captioner = auto_captioner
        self.caption_settings_form = auto_captioner.caption_settings_form
        self.image_list = auto_captioner.image_list
        self.image_list_model = auto_captioner.image_list_model
        self.settings = get_settings()
        self.captioning_thread = None
        self.steps: list[dict] = []
        self.step_index = 0
        self.selected_image_indices: list = []
        self.setWindowTitle('Model Chains')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(500, 500)
        layout = QVBoxLayout(self)

        chain_row_layout = QHBoxLayout()
        self.chain_combo_box = QComboBox()
        self.chain_combo_box.setEditable(False)
        self.save_chain_button = QPushButton('Save As...')
        self.delete_chain_button = QPushButton('Delete')
        chain_row_layout.addWidget(self.chain_combo_box)
        chain_row_layout.addWidget(self.save_chain_button)
        chain_row_layout.addWidget(self.delete_chain_button)
        layout.addLayout(chain_row_layout)

        self.steps_list_widget = QListWidget()
        layout.addWidget(self.steps_list_widget)

        step_buttons_layout = QHBoxLayout()
        self.add_step_button = QPushButton('Add Current Settings as Step')
        self.add_step_button.setToolTip(
            'Append a snapshot of every current Auto-Captioner setting '
            '(model, prompt, generation parameters, etc.) as the next '
            'step in this chain.')
        self.remove_step_button = QPushButton('Remove Step')
        self.move_step_up_button = QPushButton('Move Up')
        self.move_step_down_button = QPushButton('Move Down')
        step_buttons_layout.addWidget(self.add_step_button)
        step_buttons_layout.addWidget(self.remove_step_button)
        step_buttons_layout.addWidget(self.move_step_up_button)
        step_buttons_layout.addWidget(self.move_step_down_button)
        layout.addLayout(step_buttons_layout)

        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 6)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        layout.addWidget(self.console_text_edit)

        run_buttons_layout = QHBoxLayout()
        self.run_stop_button = QPushButton('Run Chain')
        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.reject)
        run_buttons_layout.addWidget(self.run_stop_button)
        run_buttons_layout.addWidget(self.close_button)
        run_buttons_layout.addStretch()
        layout.addLayout(run_buttons_layout)

        self.chain_combo_box.currentTextChanged.connect(self.load_chain)
        self.save_chain_button.clicked.connect(self.save_chain)
        self.delete_chain_button.clicked.connect(self.delete_chain)
        self.add_step_button.clicked.connect(self.add_step)
        self.remove_step_button.clicked.connect(self.remove_step)
        self.move_step_up_button.clicked.connect(
            lambda: self.move_step(-1))
        self.move_step_down_button.clicked.connect(
            lambda: self.move_step(1))
        self.run_stop_button.clicked.connect(self.start_or_stop)

        self.reload_chains()

    def refresh_steps_list(self):
        self.steps_list_widget.clear()
        for index, step_settings in enumerate(self.steps):
            self.steps_list_widget.addItem(
                describe_step(step_settings, index + 1))

    @Slot()
    def add_step(self):
        self.steps.append(self.caption_settings_form.get_caption_settings())
        self.refresh_steps_list()
        self.steps_list_widget.setCurrentRow(len(self.steps) - 1)

    @Slot()
    def remove_step(self):
        row = self.steps_list_widget.currentRow()
        if row < 0:
            return
        del self.steps[row]
        self.refresh_steps_list()

    def move_step(self, direction: int):
        row = self.steps_list_widget.currentRow()
        new_row = row + direction
        if row < 0 or not 0 <= new_row < len(self.steps):
            return
        self.steps[row], self.steps[new_row] = (self.steps[new_row],
                                                 self.steps[row])
        self.refresh_steps_list()
        self.steps_list_widget.setCurrentRow(new_row)

    def reload_chains(self):
        self.chain_combo_box.blockSignals(True)
        self.chain_combo_box.clear()
        self.chain_combo_box.addItem('')
        self.chain_combo_box.addItems(list(get_model_chains()))
        self.chain_combo_box.blockSignals(False)

    @Slot()
    def save_chain(self):
        if not self.steps:
            QMessageBox.warning(self, 'Model Chains',
                                'Add at least one step first.')
            return
        name, is_confirmed = QInputDialog.getText(
            self, 'Save Model Chain', 'Chain name:')
        name = name.strip()
        if not is_confirmed or not name:
            return
        chains = get_model_chains()
        if name in chains:
            reply = QMessageBox.question(
                self, 'Save Model Chain',
                f'A chain named "{name}" already exists. Overwrite it?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        chains[name] = self.steps
        save_model_chains(chains)
        self.reload_chains()
        self.chain_combo_box.blockSignals(True)
        self.chain_combo_box.setCurrentText(name)
        self.chain_combo_box.blockSignals(False)

    @Slot(str)
    def load_chain(self, name: str):
        if not name:
            return
        chains = get_model_chains()
        steps = chains.get(name)
        if steps is None:
            return
        self.steps = steps
        self.refresh_steps_list()

    @Slot()
    def delete_chain(self):
        name = self.chain_combo_box.currentText()
        if not name:
            return
        chains = get_model_chains()
        if name not in chains:
            return
        reply = QMessageBox.question(
            self, 'Delete Model Chain',
            f'Delete the model chain "{name}"?')
        if reply != QMessageBox.StandardButton.Yes:
            return
        del chains[name]
        save_model_chains(chains)
        self.reload_chains()

    @Slot()
    def start_or_stop(self):
        if self.captioning_thread is not None:
            self.cancel_current_run()
            self.run_stop_button.setEnabled(False)
            self.run_stop_button.setText('Canceling...')
            return
        if not self.steps:
            QMessageBox.warning(self, 'Model Chains',
                                'Add at least one step first.')
            return
        selected_image_indices = self.image_list.get_selected_image_indices()
        if not selected_image_indices:
            QMessageBox.warning(self, 'Model Chains',
                                'Select at least one image first.')
            return
        selected_image_count = len(selected_image_indices)
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.selected_image_indices = selected_image_indices
        # Don't let main-window captioning race with the chain run on the
        # shared model cache.
        self.auto_captioner.start_cancel_button.setEnabled(False)
        self.console_text_edit.show()
        self.console_text_edit.clear()
        self.close_button.setEnabled(False)
        self.run_stop_button.setText('Cancel Chain')
        self.step_index = 0
        self.start_step()

    def start_step(self):
        step_settings = self.steps[self.step_index]
        selected_image_count = len(self.selected_image_indices)
        if step_settings.get(
                'caption_position', CaptionPosition.DO_NOT_ADD
        ) != CaptionPosition.DO_NOT_ADD:
            self.image_list_model.add_to_undo_stack(
                action_name=f'Run Model Chain Step {self.step_index + 1} '
                            f'({pluralize("Image", selected_image_count)})',
                should_ask_for_confirmation=selected_image_count > 1)
        tag_separator = get_tag_separator()
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        models_directory_path = (Path(models_directory_path)
                                 if models_directory_path else None)
        self.captioning_thread = CaptioningThread(
            self.auto_captioner, self.image_list_model,
            self.selected_image_indices, step_settings, tag_separator,
            models_directory_path)
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            self.auto_captioner.caption_generated)
        self.captioning_thread.finished.connect(
            self.step_finished, Qt.ConnectionType.QueuedConnection)
        # Redirect `stdout`/`stderr` to the thread so its `print()`-based
        # progress messages reach `text_outputted` (and therefore the
        # console) at all, the same way `AutoCaptioner.generate_captions()`
        # does - restored once the whole chain finishes or is canceled, in
        # `finish_run()`.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()

    def update_console_text_edit(self, text: str):
        text = text.strip()
        if not text:
            return
        self.console_text_edit.appendPlainText(
            f'[Step {self.step_index + 1}] {text}')

    @Slot()
    def step_finished(self):
        thread = self.captioning_thread
        self.captioning_thread = None
        if thread.is_canceled:
            self.finish_run()
            return
        if thread.is_error:
            reply = QMessageBox.question(
                self, 'Model Chains',
                f'Step {self.step_index + 1} failed for every image (see '
                f'the console above for the error). Continue with the '
                f'next step anyway?')
            if reply != QMessageBox.StandardButton.Yes:
                self.finish_run()
                return
        self.step_index += 1
        if self.step_index < len(self.steps):
            self.start_step()
            return
        self.finish_run()

    def finish_run(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.close_button.setEnabled(True)
        self.run_stop_button.setEnabled(True)
        self.run_stop_button.setText('Run Chain')
        self.auto_captioner.start_cancel_button.setEnabled(True)

    def cancel_current_run(self):
        if self.captioning_thread is None:
            return
        thread = self.captioning_thread
        # Disconnect before canceling so that late signals can't touch the
        # closing dialog; the thread itself stops at its next cancel check.
        thread.disconnect()
        thread.is_canceled = True
        self.captioning_thread = None
        self.finish_run()

    def closeEvent(self, event):
        self.cancel_current_run()
        super().closeEvent(event)
