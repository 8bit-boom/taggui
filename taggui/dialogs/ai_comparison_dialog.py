import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from auto_captioning.ai_comparison import (build_candidates, build_judge_prompt,
                                           detect_unsloth_endpoint,
                                           judge_captions, parse_judge_response)
from auto_captioning import local_llama
from auto_captioning.captioning_thread import CaptioningThread
from auto_captioning.generation_presets import GENERATION_PRESETS
from auto_captioning.prompt_templates import (PROMPT_TEMPLATES,
                                              PROMPT_TEMPLATE_PURPOSES)
from utils.settings import get_tag_separator
from utils.utils import set_text_edit_height


class AIComparisonDialog(QDialog):
    """
    One-button AI comparison of command prompts and generation settings.

    Builds candidate configurations (prompt candidates x settings candidates),
    captions a few sample images with each (without applying them), then asks a
    judge LLM on a local OpenAI-compatible server (Ollama / vLLM / llama.cpp,
    e.g. Qwen3.8-27B or Gemma 4) to score and rank them. Shows a ranked report
    and can apply the winning prompt/settings to the Auto-Captioner.
    """

    judge_finished = Signal(str, str)  # (response_text, error_message)

    def __init__(self, auto_captioner):
        super().__init__(None)
        self.auto_captioner = auto_captioner
        self.caption_settings_form = auto_captioner.caption_settings_form
        self.image_list = auto_captioner.image_list
        self.image_list_model = auto_captioner.image_list_model
        self.captioning_thread = None
        self.candidates: list[dict] = []
        self.candidate_index = 0
        self.sample_indices: list = []
        self.image_order: list[tuple[str, int]] = []
        self.captions: dict[tuple[str, int], dict[str, str]] = {}
        self.verdict: dict | None = None
        self.setWindowTitle('AI Compare Prompts & Settings')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(820, 720)
        layout = QVBoxLayout(self)

        # --- Candidate selection -------------------------------------------
        candidates_group = QGroupBox('Candidates to compare')
        candidates_layout = QVBoxLayout(candidates_group)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel('Prompt templates for:'))
        self.target_model_combo = QComboBox()
        self.target_model_combo.addItems(list(PROMPT_TEMPLATES))
        target_row.addWidget(self.target_model_combo)
        self.include_current_prompt_check = QCheckBox('Include current prompt')
        self.include_current_prompt_check.setChecked(True)
        target_row.addWidget(self.include_current_prompt_check)
        target_row.addStretch()
        candidates_layout.addLayout(target_row)
        purposes_row = QHBoxLayout()
        purposes_row.addWidget(QLabel('Purposes:'))
        self.purpose_checks: dict[str, QCheckBox] = {}
        for purpose in PROMPT_TEMPLATE_PURPOSES:
            check = QCheckBox(purpose)
            check.setChecked(purpose in ('General', 'Character', 'Style'))
            self.purpose_checks[purpose] = check
            purposes_row.addWidget(check)
        purposes_row.addStretch()
        candidates_layout.addLayout(purposes_row)
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel('Generation presets:'))
        self.preset_checks: dict[str, QCheckBox] = {}
        for name in GENERATION_PRESETS:
            check = QCheckBox(name)
            self.preset_checks[name] = check
            settings_row.addWidget(check)
        settings_row.addWidget(QLabel('(current settings always included)'))
        settings_row.addStretch()
        candidates_layout.addLayout(settings_row)
        images_row = QHBoxLayout()
        images_row.addWidget(QLabel('Sample images (first selected):'))
        self.image_count_spin = QSpinBox()
        self.image_count_spin.setRange(1, 50)
        self.image_count_spin.setValue(min(3, max(
            1, len(self.image_list.get_selected_image_indices()))))
        images_row.addWidget(self.image_count_spin)
        self.candidate_count_label = QLabel('')
        images_row.addWidget(self.candidate_count_label)
        images_row.addStretch()
        candidates_layout.addLayout(images_row)
        layout.addWidget(candidates_group)

        # --- Judge configuration -------------------------------------------
        judge_group = QGroupBox('AI judge')
        judge_layout = QVBoxLayout(judge_group)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel('Judge source:'))
        self.judge_source_combo = QComboBox()
        self.judge_source_combo.addItems(
            ['Local GGUF (in-app, no server)', 'OpenAI server (URL)'])
        source_row.addWidget(self.judge_source_combo)
        source_row.addStretch()
        judge_layout.addLayout(source_row)

        # Local in-app GGUF widgets (no server / URL / API key needed).
        self.local_judge_widget = QWidget()
        local_form = QFormLayout(self.local_judge_widget)
        self.gguf_combo = QComboBox()
        self.gguf_combo.setEditable(True)
        self.gguf_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.gguf_combo.setMinimumContentsLength(20)
        gguf_row = QHBoxLayout()
        gguf_row.addWidget(self.gguf_combo, stretch=1)
        self.browse_gguf_button = QPushButton('Browse...')
        self.browse_gguf_button.clicked.connect(self.browse_gguf)
        gguf_row.addWidget(self.browse_gguf_button)
        local_form.addRow('GGUF model', gguf_row)
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(-1, 999)
        self.gpu_layers_spin.setValue(-1)
        self.gpu_layers_spin.setToolTip('-1 offloads all layers to the GPU.')
        local_form.addRow('GPU layers', self.gpu_layers_spin)
        self.local_ctx_spin = QSpinBox()
        self.local_ctx_spin.setRange(512, 131072)
        self.local_ctx_spin.setValue(8192)
        local_form.addRow('Context length', self.local_ctx_spin)
        judge_layout.addWidget(self.local_judge_widget)

        # OpenAI-server widgets (URL / model / key, e.g. Unsloth / vLLM).
        self.server_judge_widget = QWidget()
        server_form = QFormLayout(self.server_judge_widget)
        self.judge_base_url_edit = QLineEdit('http://127.0.0.1:8000/v1')
        self.judge_model_edit = QLineEdit()
        self.judge_model_edit.setPlaceholderText(
            'e.g. qwen3.8:27b / gemma-4 (blank = auto-detect)')
        self.judge_api_key_edit = QLineEdit()
        self.judge_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        server_form.addRow('Server URL', self.judge_base_url_edit)
        server_form.addRow('Judge model', self.judge_model_edit)
        server_form.addRow('API key (optional)', self.judge_api_key_edit)
        self.detect_unsloth_button = QPushButton('Detect Unsloth Studio')
        self.detect_unsloth_button.setToolTip(
            'Auto-fill the URL, model and API key from the model currently '
            'loaded in Unsloth Studio (reads its local logs).')
        self.detect_unsloth_button.clicked.connect(self.detect_unsloth)
        server_form.addRow('', self.detect_unsloth_button)
        judge_layout.addWidget(self.server_judge_widget)
        layout.addWidget(judge_group)
        self.judge_source_combo.currentTextChanged.connect(
            self.update_judge_source_visibility)

        # --- Buttons -------------------------------------------------------
        self.run_button = QPushButton('Run AI Comparison')
        self.run_button.clicked.connect(self.start_or_stop)
        self.apply_winner_button = QPushButton('Apply Winner')
        self.apply_winner_button.setEnabled(False)
        self.apply_winner_button.clicked.connect(self.apply_winner)
        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.apply_winner_button)
        buttons.addWidget(self.close_button)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 5)
        self.console_text_edit.setReadOnly(True)
        layout.addWidget(self.console_text_edit)

        self.verdict_text_edit = QPlainTextEdit()
        set_text_edit_height(self.verdict_text_edit, 6)
        self.verdict_text_edit.setReadOnly(True)
        self.verdict_text_edit.setPlaceholderText(
            'The AI ranking and reasoning will appear here.')
        layout.addWidget(self.verdict_text_edit)

        self.results_table = QTableWidget()
        self.results_table.setWordWrap(True)
        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        layout.addWidget(self.results_table, stretch=1)

        self.judge_finished.connect(self.on_judge_finished)
        self.target_model_combo.currentTextChanged.connect(
            self.update_candidate_count)
        for check in (*self.purpose_checks.values(),
                      *self.preset_checks.values()):
            check.toggled.connect(self.update_candidate_count)
        self.include_current_prompt_check.toggled.connect(
            self.update_candidate_count)
        self.update_candidate_count()
        # Best-effort: pre-fill the server judge from a loaded Unsloth model.
        endpoint = detect_unsloth_endpoint()
        if endpoint is not None:
            self.judge_base_url_edit.setText(endpoint['base_url'])
            self.judge_model_edit.setText(endpoint['model'])
            self.judge_api_key_edit.setText(endpoint['api_key'])
        # Populate the in-app GGUF list and prefer it when a model is found,
        # so the judge works with nothing but this app (no server needed).
        self.refresh_gguf_list()
        if local_llama.is_available() and self.gguf_combo.count() > 0:
            self.judge_source_combo.setCurrentText(
                'Local GGUF (in-app, no server)')
        self.update_judge_source_visibility(
            self.judge_source_combo.currentText())

    # --- Judge source ------------------------------------------------------
    @Slot(str)
    def update_judge_source_visibility(self, source: str):
        is_local = source.startswith('Local GGUF')
        self.local_judge_widget.setVisible(is_local)
        self.server_judge_widget.setVisible(not is_local)

    @Slot()
    def refresh_gguf_list(self):
        current = self.gguf_combo.currentText()
        self.gguf_combo.clear()
        models_dir = self.caption_settings_form.settings.value(
            'models_directory_path', defaultValue='', type=str)
        extra = [Path(models_dir)] if models_dir else []
        for path in local_llama.discover_gguf_models(extra):
            self.gguf_combo.addItem(str(path))
        if current:
            self.gguf_combo.setCurrentText(current)

    @Slot()
    def browse_gguf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a GGUF model', '', 'GGUF models (*.gguf)')
        if path:
            self.gguf_combo.setCurrentText(path)

    # --- Candidate preview -------------------------------------------------
    def _prompt_candidates(self) -> list[str]:
        prompts = []
        if self.include_current_prompt_check.isChecked():
            current = self.caption_settings_form.prompt_text_edit.toPlainText()
            if current.strip():
                prompts.append(current)
        model = self.target_model_combo.currentText()
        for purpose, check in self.purpose_checks.items():
            if check.isChecked():
                prompts.append(PROMPT_TEMPLATES[model][purpose])
        # De-duplicate while preserving order.
        seen = set()
        unique = []
        for prompt in prompts:
            if prompt not in seen:
                seen.add(prompt)
                unique.append(prompt)
        return unique

    def _settings_candidates(self) -> list[dict]:
        base = self.caption_settings_form.get_caption_settings()
        candidates = [{'name': 'Current settings',
                       'generation_parameters':
                           base.get('generation_parameters', {})}]
        for name, check in self.preset_checks.items():
            if check.isChecked():
                candidates.append({'name': name,
                                   'generation_parameters':
                                       dict(GENERATION_PRESETS[name])})
        return candidates

    @Slot()
    def update_candidate_count(self):
        count = len(self._prompt_candidates()) * len(self._settings_candidates())
        self.candidate_count_label.setText(f'{count} candidate configurations')

    @Slot()
    def detect_unsloth(self):
        endpoint = detect_unsloth_endpoint()
        if endpoint is None:
            self.log('Unsloth Studio not detected. Load a model in Unsloth '
                     'Studio (e.g. Qwen3.8-27B), then press Detect again, or '
                     'enter the OpenAI SDK base URL and API key it shows.')
            return
        self.judge_base_url_edit.setText(endpoint['base_url'])
        self.judge_model_edit.setText(endpoint['model'])
        self.judge_api_key_edit.setText(endpoint['api_key'])
        self.log(f'Detected Unsloth Studio judge: {endpoint["model"]} at '
                 f'{endpoint["base_url"]}')

    def log(self, text: str):
        self.console_text_edit.appendPlainText(text)

    # --- Run orchestration -------------------------------------------------
    @Slot()
    def start_or_stop(self):
        if self.captioning_thread is not None:
            self.cancel_current_run()
            return
        selected = self.image_list.get_selected_image_indices()
        if not selected:
            QMessageBox.warning(self, 'AI Compare',
                                'Select at least one image first.')
            return
        base_settings = self.caption_settings_form.get_caption_settings()
        self.candidates = build_candidates(self._prompt_candidates(),
                                           self._settings_candidates(),
                                           base_settings)
        if not self.candidates:
            QMessageBox.warning(self, 'AI Compare',
                                'No candidates to compare. Add a prompt or '
                                'select a purpose/preset.')
            return
        self.sample_indices = selected[:self.image_count_spin.value()]
        self.candidate_index = 0
        self.image_order = []
        self.captions = {}
        self.verdict = None
        self.verdict_text_edit.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self.apply_winner_button.setEnabled(False)
        self.auto_captioner.start_cancel_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.run_button.setText('Cancel Comparison')
        self.log(f'Comparing {len(self.candidates)} candidates on '
                 f'{len(self.sample_indices)} images...')
        self.start_candidate_run()

    def start_candidate_run(self):
        candidate = self.candidates[self.candidate_index]
        self.log(f'[{candidate["label"]}] {candidate["description"]}')
        self.captioning_thread = CaptioningThread(
            self.auto_captioner, self.image_list_model, self.sample_indices,
            candidate['settings'], get_tag_separator(), None)
        self.captioning_thread.text_outputted.connect(self.log)
        self.captioning_thread.caption_generated.connect(
            lambda index, caption, tags, label=candidate['label']:
            self.collect_caption(index, caption, label))
        self.captioning_thread.finished.connect(
            self.candidate_finished, Qt.ConnectionType.QueuedConnection)
        self.captioning_thread.start()

    def collect_caption(self, image_index, caption: str, label: str):
        image = self.image_list_model.data(image_index,
                                           Qt.ItemDataRole.UserRole)
        if image is None:
            return
        key = (image.path.name, image_index.row())
        if key not in self.captions:
            self.captions[key] = {}
            self.image_order.append(key)
        self.captions[key][label] = caption

    @Slot()
    def candidate_finished(self):
        thread = self.captioning_thread
        self.captioning_thread = None
        if thread is not None and thread.is_canceled:
            self.finish_run()
            return
        self.candidate_index += 1
        if self.candidate_index < len(self.candidates):
            self.start_candidate_run()
            return
        self.start_judging()

    def start_judging(self):
        results = [{'image': name, 'captions': self.captions[(name, row)]}
                   for (name, row) in self.image_order]
        target = self.target_model_combo.currentText()
        judge_prompt = build_judge_prompt(self.candidates, results,
                                          f'{target} training captions')
        is_local = self.judge_source_combo.currentText().startswith('Local GGUF')
        if is_local:
            model_path = self.gguf_combo.currentText().strip()
            n_gpu_layers = self.gpu_layers_spin.value()
            n_ctx = self.local_ctx_spin.value()
            self.log(f'Asking the in-app local judge ({Path(model_path).name}) '
                     f'to rank the candidates...')

            def worker():
                try:
                    text = local_llama.judge_captions_local(
                        model_path, judge_prompt, n_gpu_layers, n_ctx)
                    self.judge_finished.emit(text, '')
                except Exception as exception:  # noqa: BLE001 - report to UI
                    self.judge_finished.emit('', str(exception))
        else:
            base_url = self.judge_base_url_edit.text().strip()
            model = self.judge_model_edit.text().strip() or None
            api_key = self.judge_api_key_edit.text().strip() or None
            self.log('Asking the AI judge to rank the candidates...')

            def worker():
                try:
                    text = judge_captions(base_url, api_key, model, judge_prompt)
                    self.judge_finished.emit(text, '')
                except Exception as exception:  # noqa: BLE001 - report to UI
                    self.judge_finished.emit('', str(exception))

        threading.Thread(target=worker, daemon=True).start()

    @Slot(str, str)
    def on_judge_finished(self, text: str, error: str):
        if error:
            self.log(f'Judge failed: {error}')
            self.verdict_text_edit.setPlainText(
                f'The AI judge could not be reached:\n{error}\n\n'
                f'The caption comparison table below still shows every '
                f'candidate\'s output so you can compare them manually.')
            self.show_results_table()
            self.finish_run()
            return
        labels = [candidate['label'] for candidate in self.candidates]
        self.verdict = parse_judge_response(text, labels)
        self.show_verdict()
        self.show_results_table()
        self.apply_winner_button.setEnabled(self.verdict['best'] is not None)
        self.finish_run()

    def show_verdict(self):
        verdict = self.verdict
        lines = ['AI ranking (best first):']
        for rank, label in enumerate(verdict['ranking'], start=1):
            candidate = next((c for c in self.candidates
                              if c['label'] == label), None)
            score = verdict['scores'].get(label, 0.0)
            description = candidate['description'] if candidate else label
            lines.append(f'  {rank}. {label} (score {score:.1f}) - {description}')
        if verdict['best']:
            lines.append(f'\nWinner: {verdict["best"]}')
        if verdict['reasoning']:
            lines.append(f'\nReasoning:\n{verdict["reasoning"]}')
        if not verdict['valid']:
            lines.append('\n(Note: the judge response was not fully '
                         'parseable; showing a fallback ranking.)')
        self.verdict_text_edit.setPlainText('\n'.join(lines))

    def show_results_table(self):
        labels = [candidate['label'] for candidate in self.candidates]
        self.results_table.setColumnCount(1 + len(labels))
        self.results_table.setHorizontalHeaderLabels(['Image'] + labels)
        self.results_table.setRowCount(len(self.image_order))
        for row, (name, _) in enumerate(self.image_order):
            self.results_table.setItem(row, 0, QTableWidgetItem(name))
            for column, label in enumerate(labels, start=1):
                caption = self.captions[(name, _)].get(label, '')
                item = QTableWidgetItem(caption)
                item.setToolTip(caption)
                self.results_table.setItem(row, column, item)
        self.results_table.resizeColumnsToContents()
        self.results_table.resizeRowsToContents()

    @Slot()
    def apply_winner(self):
        if not self.verdict or not self.verdict['best']:
            return
        winner = next((c for c in self.candidates
                       if c['label'] == self.verdict['best']), None)
        if winner is None:
            return
        # Apply only the prompt and generation parameters; leave the caption
        # position and other run settings as the user had them.
        self.caption_settings_form.apply_caption_settings({
            'prompt': winner['prompt'],
            'generation_parameters':
                winner['settings']['generation_parameters'],
        })
        self.log(f'Applied winner {winner["label"]}: {winner["description"]}')

    def finish_run(self):
        self.captioning_thread = None
        self.run_button.setEnabled(True)
        self.run_button.setText('Run AI Comparison')
        self.close_button.setEnabled(True)
        self.auto_captioner.start_cancel_button.setEnabled(True)

    def cancel_current_run(self):
        thread = self.captioning_thread
        if thread is not None:
            thread.disconnect()
            thread.is_canceled = True
            self.captioning_thread = None
        self.finish_run()

    def closeEvent(self, event):
        self.cancel_current_run()
        super().closeEvent(event)
