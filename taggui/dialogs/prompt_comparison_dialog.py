from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from auto_captioning.captioning_thread import CaptioningThread
from utils.enums import CaptionPosition
from utils.settings import get_tag_separator
from utils.utils import set_text_edit_height


class PromptComparisonDialog(QDialog):
    """
    Caption a few selected images with two different prompts (A/B) without
    applying the captions, and show the outputs side by side. Runs the
    same `CaptioningThread` machinery as normal captioning, but with the
    caption position set to `Do not add to tags`, so no image list model is
    modified and no undo stack entry is created. The thread is parented to
    the `AutoCaptioner`, so its loaded model is reused instead of loaded
    again.
    """

    def __init__(self, auto_captioner):
        super().__init__(None)
        self.auto_captioner = auto_captioner
        self.caption_settings_form = auto_captioner.caption_settings_form
        self.image_list = auto_captioner.image_list
        self.image_list_model = auto_captioner.image_list_model
        self.captioning_thread = None
        self.prompt_index = 0
        # {(image name, row): [prompt A caption, prompt B caption]}
        self.captions: dict[tuple[str, int], list[str]] = {}
        self.prompts: list[str] = []
        self.setWindowTitle('Compare Prompts')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        layout = QVBoxLayout(self)

        settings_form = QFormLayout()
        settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.prompt_a_text_edit = QPlainTextEdit()
        self.prompt_b_text_edit = QPlainTextEdit()
        for text_edit in (self.prompt_a_text_edit, self.prompt_b_text_edit):
            set_text_edit_height(text_edit, 4)
        current_prompt = (self.caption_settings_form.prompt_text_edit
                          .toPlainText())
        self.prompt_a_text_edit.setPlainText(current_prompt)
        self.prompt_b_text_edit.setPlaceholderText(
            'Enter a second prompt to compare against')
        self.image_count_spin_box = QSpinBox()
        self.image_count_spin_box.setMinimum(1)
        self.image_count_spin_box.setMaximum(100)
        self.image_count_spin_box.setValue(min(5, max(
            1, len(self.image_list.get_selected_image_indices()))))
        settings_form.addRow('Prompt A', self.prompt_a_text_edit)
        settings_form.addRow('Prompt B', self.prompt_b_text_edit)
        settings_form.addRow('Images (first selected)',
                             self.image_count_spin_box)
        layout.addLayout(settings_form)

        self.start_stop_button = QPushButton('Start Comparison')
        self.start_stop_button.clicked.connect(self.start_or_stop)
        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.reject)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_stop_button)
        buttons_layout.addWidget(self.close_button)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 6)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        layout.addWidget(self.console_text_edit)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(
            ['Image', 'Prompt A', 'Prompt B'])
        self.results_table.setWordWrap(True)
        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.hide()
        layout.addWidget(self.results_table)

    @Slot()
    def start_or_stop(self):
        if self.captioning_thread is not None:
            self.cancel_current_run()
            self.start_stop_button.setEnabled(False)
            self.start_stop_button.setText('Canceling...')
            return
        selected_image_indices = self.image_list.get_selected_image_indices()
        if not selected_image_indices:
            QMessageBox.warning(self, 'Compare Prompts',
                                'Select at least one image first.')
            return
        prompt_a = self.prompt_a_text_edit.toPlainText()
        prompt_b = self.prompt_b_text_edit.toPlainText()
        if not prompt_b.strip():
            QMessageBox.warning(self, 'Compare Prompts',
                                'Enter a second prompt to compare against.')
            return
        self.prompts = [prompt_a, prompt_b]
        image_count = self.image_count_spin_box.value()
        sample_indices = selected_image_indices[:image_count]
        # Don't let main-window captioning race with the comparison runs on
        # the shared model cache.
        self.auto_captioner.start_cancel_button.setEnabled(False)
        self.console_text_edit.show()
        self.console_text_edit.clear()
        self.results_table.hide()
        self.results_table.setRowCount(0)
        self.captions = {}
        self.prompt_index = 0
        self.close_button.setEnabled(False)
        self.start_stop_button.setText('Cancel Comparison')
        self.start_next_run(sample_indices)

    def start_next_run(self, sample_indices):
        prompt_label = f'Prompt {chr(ord("A") + self.prompt_index)}'
        caption_settings = self.caption_settings_form.get_caption_settings()
        # Captions must not be applied to the image list; the skip/test-run
        # settings are run-management options, not comparison options.
        caption_settings['caption_position'] = CaptionPosition.DO_NOT_ADD
        caption_settings['prompt'] = self.prompts[self.prompt_index]
        caption_settings['skip_already_captioned'] = False
        caption_settings['test_run_image_count'] = 0
        self.captioning_thread = CaptioningThread(
            self.auto_captioner, self.image_list_model, sample_indices,
            caption_settings, get_tag_separator(), None)
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            lambda index, caption, tags, label_=prompt_label:
            self.collect_caption(index, caption, label_))
        self.captioning_thread.finished.connect(
            self.run_finished, Qt.ConnectionType.QueuedConnection)
        self.captioning_thread.start()

    def collect_caption(self, image_index, caption: str, prompt_label: str):
        image = self.image_list_model.data(image_index,
                                           Qt.ItemDataRole.UserRole)
        if image is None:
            return
        key = (image.path.name, image_index.row())
        if key not in self.captions:
            self.captions[key] = ['', '']
        column = 0 if prompt_label == 'Prompt A' else 1
        self.captions[key][column] = caption

    def update_console_text_edit(self, text: str):
        text = text.strip()
        if not text:
            return
        self.console_text_edit.appendPlainText(text)

    @Slot()
    def run_finished(self):
        self.prompt_index += 1
        if self.prompt_index < len(self.prompts):
            # Run the second prompt on the same images.
            selected_image_indices = (
                self.image_list.get_selected_image_indices())
            image_count = self.image_count_spin_box.value()
            self.captioning_thread = None
            self.start_next_run(selected_image_indices[:image_count])
            return
        self.captioning_thread = None
        self.finish_run()

    def finish_run(self):
        self.close_button.setEnabled(True)
        self.start_stop_button.setEnabled(True)
        self.start_stop_button.setText('Start Comparison')
        self.auto_captioner.start_cancel_button.setEnabled(True)
        self.show_results_table()

    def cancel_current_run(self):
        if self.captioning_thread is None:
            return
        thread = self.captioning_thread
        # Disconnect before canceling so that late signals can't touch the
        # closing dialog; the thread itself stops at its next cancel check
        # (immediately for local models, via the stopping criteria).
        thread.disconnect()
        thread.is_canceled = True
        self.captioning_thread = None
        self.finish_run()

    def closeEvent(self, event):
        self.cancel_current_run()
        super().closeEvent(event)

    def show_results_table(self):
        self.results_table.show()
        rows = sorted(self.captions.items(), key=lambda item: item[0][1])
        self.results_table.setRowCount(len(rows))
        for row_index, ((image_name, _), captions) in enumerate(rows):
            for column, text in enumerate([image_name, captions[0],
                                           captions[1]]):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.results_table.setItem(row_index, column, item)
        self.results_table.resizeColumnsToContents()
        self.results_table.resizeRowsToContents()
