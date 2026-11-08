import gc
import sys
from pathlib import Path

import torch
from PySide6.QtCore import QModelIndex, Qt, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QAbstractScrollArea, QComboBox, QDockWidget,
                               QFormLayout, QFrame, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from auto_captioning.load_presets import LOAD_PRESETS
from auto_captioning.caption_profiles import (get_caption_profiles,
                                              save_caption_profiles)
from auto_captioning.captioning_thread import CaptioningThread
from auto_captioning.model_settings import (get_model_settings,
                                            save_model_settings)
from auto_captioning.auto_captioning_model import ATTENTION_IMPLEMENTATIONS
from auto_captioning.generation_presets import GENERATION_PRESETS
from auto_captioning.prompt_templates import (CAPTIONER_PROMPT_SUFFIXES,
                                              PROMPT_TEMPLATE_CAPTIONERS,
                                              PROMPT_TEMPLATE_PURPOSES,
                                              PROMPT_TEMPLATES)
from auto_captioning.models.qwen3_vl import (PIXELS_PER_VISUAL_TOKEN,
                                             Qwen3VL)
from auto_captioning.models.wd_tagger import WdTagger
from auto_captioning.models_list import (REMOTE_ENDPOINT_MODEL_ID,
                                         get_all_models, get_model_class,
                                         is_remote_endpoint_model)
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from dialogs.model_manager_dialog import ModelManagerDialog
from dialogs.model_chain_dialog import ModelChainDialog
from dialogs.prompt_comparison_dialog import PromptComparisonDialog
from models.image_list_model import ImageListModel
from utils.big_widgets import TallPushButton
from utils.enums import CaptionDevice, CaptionPosition
from utils.settings import DEFAULT_SETTINGS, get_settings, get_tag_separator
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsDoubleSpinBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsPlainTextEdit)
from utils.utils import pluralize, set_text_edit_height
from widgets.image_list import ImageList


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Raised)


def create_help_label(tooltip: str) -> QLabel:
    """
    A small "?" label that shows `tooltip` on hover, for appending next to
    a setting's name to explain what it does.
    """
    help_label = QLabel('(?)')
    help_label.setToolTip(tooltip)
    help_label.setStyleSheet('color: palette(link);')
    help_label.setCursor(Qt.CursorShape.WhatsThisCursor)
    return help_label


def label_with_help(text: str, tooltip: str) -> QWidget:
    """
    A form-row label: `text` followed by a help icon that shows `tooltip`
    on hover.
    """
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(QLabel(text))
    layout.addWidget(create_help_label(tooltip))
    return container


class CaptionSettingsForm(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        try:
            import bitsandbytes
            self.is_bitsandbytes_available = True
        # `ImportError` when the package is missing, `RuntimeError` when it
        # is installed but cannot initialize (e.g. no CUDA library found).
        except (ImportError, RuntimeError):
            self.is_bitsandbytes_available = False
        basic_settings_form = QFormLayout()
        basic_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        basic_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.basic_settings_form = basic_settings_form
        # `profile_combo_box` is a plain (not settings-backed) combo box
        # because the selected profile itself is transient; applying it
        # updates the individual settings widgets below, which do persist.
        self.profile_combo_box = QComboBox()
        self.profile_combo_box.setEditable(False)
        # See the identical call on `model_combo_box` below for why this
        # matters: an unusually long profile name would otherwise
        # permanently widen the whole form.
        self.profile_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.profile_combo_box.setMinimumContentsLength(20)
        self.profile_save_button = QPushButton('Save...')
        self.profile_save_button.clicked.connect(self.save_profile)
        self.profile_delete_button = QPushButton('Delete')
        self.profile_delete_button.clicked.connect(self.delete_profile)
        # A combo box plus one or more buttons on the same row can overflow
        # the dock's width when it's tabbed into a narrow panel (the dock's
        # `QScrollArea` can't shrink buttons below their minimum size, so
        # the row gets clipped instead of wrapping). Stack the buttons on
        # their own row below the combo box instead, so they're never cut
        # off regardless of dock width.
        self.profile_row_container = QWidget()
        profile_row_layout = QVBoxLayout(self.profile_row_container)
        profile_row_layout.setContentsMargins(0, 0, 0, 0)
        profile_row_layout.addWidget(self.profile_combo_box)
        profile_buttons_layout = QHBoxLayout()
        profile_buttons_layout.addWidget(self.profile_save_button)
        profile_buttons_layout.addWidget(self.profile_delete_button)
        profile_row_layout.addLayout(profile_buttons_layout)
        basic_settings_form.addRow(
            label_with_help(
                'Profile',
                'Save or load a full snapshot of every setting on this '
                'form.'),
            self.profile_row_container)
        self.model_combo_box = FocusedScrollSettingsComboBox(key='model_id')
        # `setEditable()` must be called before `addItems()` to preserve any
        # custom model that was set.
        self.model_combo_box.setEditable(True)
        self.model_combo_box.addItems(self.get_local_model_paths())
        self.model_combo_box.addItems(get_all_models())
        self.model_combo_box.addItems([REMOTE_ENDPOINT_MODEL_ID])
        # A `QComboBox`'s default `SizeAdjustPolicy` (`AdjustToContentsOnFirstShow`)
        # locks in a minimum width based on whatever text happens to be
        # current the first time it's shown, and never shrinks back down
        # afterward. A locally-installed model's path can be very long
        # (e.g. `C:\models\...\snapshots\<hash>`), which then permanently
        # forces the whole settings form wider than the dock, regardless of
        # dock/window size. Cap the box's natural width to a fixed character
        # count instead so long paths just get truncated in the display.
        self.model_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo_box.setMinimumContentsLength(20)
        self.model_row_container = QWidget()
        model_row_layout = QVBoxLayout(self.model_row_container)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.manage_models_button = QPushButton('Manage...')
        self.manage_models_button.clicked.connect(self.show_model_manager)
        self.unload_model_button = QPushButton('Unload Model')
        self.unload_model_button.setToolTip(
            'Remove the currently loaded model from memory and free its '
            'VRAM. It will be loaded again the next time you caption.')
        self.unload_model_button.setEnabled(False)
        # (Connected by `AutoCaptioner`, which owns the model cache state.)
        model_row_layout.addWidget(self.model_combo_box)
        model_buttons_layout = QHBoxLayout()
        model_buttons_layout.addWidget(self.manage_models_button)
        model_buttons_layout.addWidget(self.unload_model_button)
        model_buttons_layout.addStretch()
        model_row_layout.addLayout(model_buttons_layout)
        # A plain (not settings-backed) pair of combo boxes: they only pick
        # which template text to insert into `prompt_text_edit` below, which
        # is itself what gets persisted.
        self.prompt_template_model_combo_box = QComboBox()
        self.prompt_template_model_combo_box.addItems(
            list(PROMPT_TEMPLATES.keys()))
        self.prompt_template_model_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.prompt_template_model_combo_box.setMinimumContentsLength(20)
        self.prompt_template_purpose_combo_box = QComboBox()
        self.prompt_template_purpose_combo_box.addItems(
            PROMPT_TEMPLATE_PURPOSES)
        self.prompt_template_purpose_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.prompt_template_purpose_combo_box.setMinimumContentsLength(20)
        self.prompt_template_captioner_combo_box = QComboBox()
        self.prompt_template_captioner_combo_box.addItems(
            PROMPT_TEMPLATE_CAPTIONERS)
        self.prompt_template_captioner_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.prompt_template_captioner_combo_box.setMinimumContentsLength(20)
        self.prompt_template_insert_button = QPushButton(
            'Insert into Prompt')
        self.prompt_template_insert_button.clicked.connect(
            self.insert_prompt_template)
        self.prompt_template_row_container = QWidget()
        prompt_template_row_layout = QVBoxLayout(
            self.prompt_template_row_container)
        prompt_template_row_layout.setContentsMargins(0, 0, 0, 0)
        prompt_template_row_layout.addWidget(
            self.prompt_template_model_combo_box)
        prompt_template_row_layout.addWidget(
            self.prompt_template_purpose_combo_box)
        prompt_template_row_layout.addWidget(
            self.prompt_template_captioner_combo_box)
        prompt_template_row_layout.addWidget(
            self.prompt_template_insert_button)
        self.prompt_text_edit = SettingsPlainTextEdit(key='prompt')
        set_text_edit_height(self.prompt_text_edit, 4)
        self.compare_prompts_button = QPushButton('Compare Prompts...')
        self.compare_prompts_button.setToolTip(
            'Caption a few selected images with two different prompts '
            'without applying the captions, to compare their output.')
        self.model_chains_button = QPushButton('Model Chains...')
        self.model_chains_button.setToolTip(
            'Build and run a sequence of captioning passes, each with '
            'its own model and settings, applied one after another to '
            'the same selected images.')
        # (Both connected by `AutoCaptioner`, which owns the captioning
        # state.)
        self.caption_start_line_edit = SettingsLineEdit(key='caption_start')
        self.caption_start_line_edit.setClearButtonEnabled(True)
        self.caption_position_combo_box = FocusedScrollSettingsComboBox(
            key='caption_position')
        self.caption_position_combo_box.addItems(list(CaptionPosition))
        # Skip images that already have tags (i.e. were captioned before),
        # so that resuming an interrupted run is one click instead of
        # re-filtering the image list by hand.
        (self.skip_already_captioned_container,
         self.skip_already_captioned_check_box) = self.create_checkbox_row(
            'Skip images that already have tags',
            key='skip_already_captioned', default=False,
            tooltip='Don\'t caption an image that already has tags. '
                   'Lets resuming an interrupted run skip straight to '
                   'the uncaptioned images.')
        # Caption only the first N images, then pause and ask whether to
        # continue - a cheap quality check before committing to hours.
        self.test_run_image_count_spin_box = FocusedScrollSettingsSpinBox(
            key='test_run_image_count', default=0, minimum=0, maximum=1000)
        self.device_combo_box = FocusedScrollSettingsComboBox(key='device')
        self.device_combo_box.addItems(list(CaptionDevice))
        # One-click recommended load/speed settings per model family
        # (see `load_presets.py`); a plain combo like the prompt-template
        # and generation-preset pickers - applying it writes to the
        # individual settings widgets, which are what gets persisted.
        self.load_preset_combo_box = QComboBox()
        self.load_preset_combo_box.addItems(list(LOAD_PRESETS.keys()))
        self.load_preset_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.load_preset_combo_box.setMinimumContentsLength(20)
        self.load_preset_apply_button = QPushButton('Apply Load Preset')
        self.load_preset_apply_button.clicked.connect(
            self.apply_load_preset)
        self.load_preset_row_container = QWidget()
        load_preset_row_layout = QVBoxLayout(self.load_preset_row_container)
        load_preset_row_layout.setContentsMargins(0, 0, 0, 0)
        load_preset_row_layout.addWidget(self.load_preset_combo_box)
        load_preset_row_layout.addWidget(self.load_preset_apply_button)
        self.load_in_4_bit_container, self.load_in_4_bit_check_box = (
            self.create_checkbox_row(
                'Load in 4-bit', key='load_in_4_bit', default=True,
                tooltip='Load the model with 4-bit quantization to use '
                       'much less VRAM, at some cost to caption quality. '
                       'Requires a compatible GPU.'))
        self.load_in_8_bit_container, self.load_in_8_bit_check_box = (
            self.create_checkbox_row(
                'Load in 8-bit', key='load_in_8_bit', default=False,
                tooltip='Load the model with 8-bit quantization to use '
                       'less VRAM than 4-bit, at a smaller cost to '
                       'caption quality. Requires a compatible GPU.'))
        self.cpu_offload_container, self.cpu_offload_check_box = (
            self.create_checkbox_row(
                'CPU offload', key='cpu_offload', default=False,
                tooltip='Automatically move parts of the model between '
                       'GPU and CPU memory to fit one too large for the '
                       'GPU alone. Slower than keeping everything on '
                       'the GPU.'))
        self.sage_attention_container, self.sage_attention_check_box = (
            self.create_checkbox_row(
                'Use SageAttention', key='sage_attention', default=False,
                tooltip='Use the SageAttention library\'s faster '
                       'attention implementation. Requires the '
                       'sageattention package and a compatible model '
                       'and GPU.'))
        self.disable_thinking_container, self.disable_thinking_check_box = (
            self.create_checkbox_row(
                'Disable thinking', key='disable_thinking', default=False,
                tooltip='Turn off this model\'s reasoning/thinking step, '
                       'if it supports one, so the caption is generated '
                       'directly instead of after visible reasoning.'))
        # The per-image pixel budget (Qwen3-VL family only): every 28x28
        # pixels becomes one visual token after the model's 2x2 patch merge,
        # so this directly controls prefill speed and VRAM - and thereby the
        # batch size that fits. The default of 1.0 MP matches the app's
        # previous hardcoded 1280 * 28 * 28 max.
        self.min_image_megapixels_spin_box = (
            FocusedScrollSettingsDoubleSpinBox(
                key='min_image_megapixels', default=0.2, minimum=0.01,
                maximum=32))
        self.min_image_megapixels_spin_box.setSingleStep(0.05)
        self.max_image_megapixels_spin_box = (
            FocusedScrollSettingsDoubleSpinBox(
                key='max_image_megapixels', default=1.0, minimum=0.01,
                maximum=32))
        self.max_image_megapixels_spin_box.setSingleStep(0.05)
        self.pixel_budget_label = QLabel('')
        # Keep the minimum below (or equal to) the maximum.
        self.min_image_megapixels_spin_box.valueChanged.connect(
            self.max_image_megapixels_spin_box.setMinimum)
        self.max_image_megapixels_spin_box.valueChanged.connect(
            self.min_image_megapixels_spin_box.setMaximum)
        self.max_image_megapixels_spin_box.valueChanged.connect(
            self.update_pixel_budget_label)
        self.update_pixel_budget_label()
        # A container so the rows can be hidden together when a non-Qwen
        # model is selected (hiding a bare `QFormLayout` field would leave
        # its row label behind).
        self.pixel_budget_settings_form_container = QWidget()
        pixel_budget_form = QFormLayout(
            self.pixel_budget_settings_form_container)
        pixel_budget_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        pixel_budget_form.addRow(
            label_with_help(
                'Min image size (MP)',
                'The smallest an image will be resized to before '
                'captioning, in megapixels.'),
            self.min_image_megapixels_spin_box)
        pixel_budget_form.addRow(
            label_with_help(
                'Max image size (MP)',
                'The largest an image will be resized to before '
                'captioning, in megapixels. Higher uses more VRAM and '
                'generates more visual tokens per image.'),
            self.max_image_megapixels_spin_box)
        pixel_budget_form.addRow('', self.pixel_budget_label)
        # Settings for the `Remote API (vLLM / SGLang)` pseudo-model, which
        # sends images to an OpenAI-compatible server instead of loading a
        # local checkpoint.
        self.remote_api_settings_form_container = QWidget()
        remote_api_settings_form = QFormLayout(
            self.remote_api_settings_form_container)
        remote_api_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        remote_api_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.api_base_url_line_edit = SettingsLineEdit(key='api_base_url')
        self.api_base_url_line_edit.setClearButtonEnabled(True)
        self.api_base_url_line_edit.setPlaceholderText(
            'http://127.0.0.1:8000/v1')
        self.api_key_line_edit = SettingsLineEdit(key='api_key')
        self.api_key_line_edit.setClearButtonEnabled(True)
        self.api_key_line_edit.setEchoMode(
            QLineEdit.EchoMode.Password)
        self.api_model_name_line_edit = SettingsLineEdit(
            key='api_model_name')
        self.api_model_name_line_edit.setClearButtonEnabled(True)
        self.api_model_name_line_edit.setPlaceholderText(
            '(auto-detected from the server)')
        remote_api_settings_form.addRow(
            label_with_help(
                'Server URL',
                'The base URL of an OpenAI-compatible server (e.g. '
                'vLLM or SGLang) to send captioning requests to, '
                'instead of loading a model locally.'),
            self.api_base_url_line_edit)
        remote_api_settings_form.addRow(
            label_with_help(
                'API key (optional)',
                'An API key to send with each request, if the server '
                'requires one.'),
            self.api_key_line_edit)
        remote_api_settings_form.addRow(
            label_with_help(
                'Server model name (optional)',
                'The model name to request from the server. Leave '
                'blank to auto-detect it from the server.'),
            self.api_model_name_line_edit)
        self.remove_tag_separators_container = QWidget()
        remove_tag_separators_layout = QHBoxLayout(
            self.remove_tag_separators_container)
        remove_tag_separators_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        remove_tag_separators_layout.setContentsMargins(0, 0, 0, 0)
        self.remove_tag_separators_check_box = SettingsBigCheckBox(
            key='remove_tag_separators', default=True)
        remove_tag_separators_label = QLabel(
            'Remove tag separators in caption')
        remove_tag_separators_layout.addWidget(remove_tag_separators_label)
        remove_tag_separators_layout.addWidget(
            create_help_label(
                'Remove the tag separator (comma by default) from '
                'generated captions, so it can\'t be confused with an '
                'actual tag boundary.'))
        remove_tag_separators_layout.addWidget(
            self.remove_tag_separators_check_box)
        basic_settings_form.addRow(
            label_with_help(
                'Model',
                'The auto-captioning model to use. Pick a built-in or '
                'locally downloaded model, or paste a Hugging Face model '
                'ID directly.'),
            self.model_row_container)
        basic_settings_form.addRow(
            label_with_help(
                'Prompt template',
                'Insert a starting-point prompt for a target image-'
                'generation model and purpose, optionally combined with '
                'instructions tuned to the selected captioning model.'),
            self.prompt_template_row_container)
        self.prompt_label = label_with_help(
            'Prompt',
            'Instructions given to the captioning model. Supports '
            '{tags}, {name}, and {directory}/{folder} template '
            'variables.')
        basic_settings_form.addRow(self.prompt_label, self.prompt_text_edit)
        basic_settings_form.addRow('', self.compare_prompts_button)
        basic_settings_form.addRow('', self.model_chains_button)
        self.caption_start_label = label_with_help(
            'Start caption with',
            'Text that every generated caption will start with.')
        basic_settings_form.addRow(self.caption_start_label,
                                   self.caption_start_line_edit)
        basic_settings_form.addRow(
            label_with_help(
                'Caption position',
                'Where to place the generated caption relative to the '
                'image\'s existing tags.'),
            self.caption_position_combo_box)
        basic_settings_form.addRow(
            self.skip_already_captioned_container)
        basic_settings_form.addRow(
            label_with_help(
                'Test run (0 = off)',
                'Caption only this many images, then pause and ask '
                'whether to continue - a cheap quality check before '
                'committing to a long run. 0 runs the full selection '
                'with no pause.'),
            self.test_run_image_count_spin_box)
        self.device_label = label_with_help(
            'Device',
            'Whether to run the model on the GPU or CPU. GPU is much '
            'faster but needs enough VRAM to fit the model.')
        basic_settings_form.addRow(self.device_label, self.device_combo_box)
        basic_settings_form.addRow(
            label_with_help(
                'Load preset',
                'Apply a set of loading and generation settings '
                '(quantization, batching, attention implementation, '
                'etc.) tuned to a specific model family.'),
            self.load_preset_row_container)
        basic_settings_form.addRow(self.load_in_4_bit_container)
        basic_settings_form.addRow(self.load_in_8_bit_container)
        basic_settings_form.addRow(self.cpu_offload_container)
        basic_settings_form.addRow(self.sage_attention_container)
        basic_settings_form.addRow(self.disable_thinking_container)
        basic_settings_form.addRow(self.pixel_budget_settings_form_container)
        basic_settings_form.addRow(self.remove_tag_separators_container)

        self.wd_tagger_settings_form_container = QWidget()
        wd_tagger_settings_form = QFormLayout(
            self.wd_tagger_settings_form_container)
        wd_tagger_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        wd_tagger_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.show_probabilities_check_box = SettingsBigCheckBox(
            key='wd_tagger_show_probabilities', default=True)
        self.min_probability_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='wd_tagger_min_probability', default=0.4, minimum=0.01,
            maximum=1)
        self.min_probability_spin_box.setSingleStep(0.01)
        self.max_tags_spin_box = FocusedScrollSettingsSpinBox(
            key='wd_tagger_max_tags', default=30, minimum=1, maximum=999)
        (self.wd_tagger_load_in_8_bit_container,
         self.wd_tagger_load_in_8_bit_check_box) = self.create_checkbox_row(
            'Load in 8-bit', key='wd_tagger_load_in_8_bit', default=False,
            tooltip='Quantize the WD Tagger model to 8-bit to use less '
                   'memory, at some cost to tagging quality.')
        tags_to_exclude_form = QFormLayout()
        tags_to_exclude_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        tags_to_exclude_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.tags_to_exclude_text_edit = SettingsPlainTextEdit(
            key='wd_tagger_tags_to_exclude')
        tags_to_exclude_form.addRow(
            label_with_help(
                'Tags to exclude',
                'Tags that should never be generated, separated by '
                'commas.'),
            self.tags_to_exclude_text_edit)
        set_text_edit_height(self.tags_to_exclude_text_edit, 4)
        wd_tagger_settings_form.addRow(
            label_with_help(
                'Show probabilities',
                'Include each tag\'s confidence score in the console '
                'output.'),
            self.show_probabilities_check_box)
        wd_tagger_settings_form.addRow(
            label_with_help(
                'Minimum probability',
                'Only include tags the model is at least this '
                'confident about, from 0 to 1.'),
            self.min_probability_spin_box)
        wd_tagger_settings_form.addRow(
            label_with_help(
                'Maximum tags',
                'The maximum number of tags to generate per image.'),
            self.max_tags_spin_box)
        wd_tagger_settings_form.addRow(self.wd_tagger_load_in_8_bit_container)
        wd_tagger_settings_form.addRow(tags_to_exclude_form)

        self.toggle_advanced_settings_form_button = TallPushButton(
            'Show Advanced Settings')

        self.advanced_settings_form_container = QWidget()
        advanced_settings_form = QFormLayout(
            self.advanced_settings_form_container)
        self.advanced_settings_form = advanced_settings_form
        advanced_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        advanced_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        bad_forced_words_form = QFormLayout()
        bad_forced_words_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        bad_forced_words_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.bad_words_line_edit = SettingsLineEdit(key='bad_words')
        self.bad_words_line_edit.setClearButtonEnabled(True)
        self.forced_words_line_edit = SettingsLineEdit(key='forced_words')
        self.forced_words_line_edit.setClearButtonEnabled(True)
        bad_forced_words_form.addRow(
            label_with_help(
                'Discourage from caption',
                'Words or phrases the model should avoid using in the '
                'caption, separated by commas.'),
            self.bad_words_line_edit)
        bad_forced_words_form.addRow(
            label_with_help(
                'Include in caption',
                'Words or phrases that should appear somewhere in the '
                'caption, separated by commas. Separate alternatives '
                'within a group with |.'),
            self.forced_words_line_edit)
        self.min_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='min_new_tokens', default=1, minimum=1, maximum=999)
        self.max_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='max_new_tokens', default=100, minimum=1, maximum=999)
        self.beam_count_spin_box = FocusedScrollSettingsSpinBox(
            key='num_beams', default=1, minimum=1, maximum=99)
        self.length_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='length_penalty', default=1, minimum=-5, maximum=5)
        self.length_penalty_spin_box.setSingleStep(0.1)
        self.use_sampling_check_box = SettingsBigCheckBox(key='do_sample',
                                                          default=False)
        # The temperature must be positive.
        self.temperature_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='temperature', default=1, minimum=0.01, maximum=2)
        self.temperature_spin_box.setSingleStep(0.01)
        self.top_k_spin_box = FocusedScrollSettingsSpinBox(
            key='top_k', default=50, minimum=0, maximum=200)
        self.top_p_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='top_p', default=1, minimum=0, maximum=1)
        self.top_p_spin_box.setSingleStep(0.01)
        self.repetition_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='repetition_penalty', default=1, minimum=1, maximum=2)
        self.repetition_penalty_spin_box.setSingleStep(0.01)
        self.no_repeat_ngram_size_spin_box = FocusedScrollSettingsSpinBox(
            key='no_repeat_ngram_size', default=3, minimum=0, maximum=5)
        self.gpu_index_spin_box = FocusedScrollSettingsSpinBox(
            key='gpu_index', default=0, minimum=0, maximum=9)
        # How many images to caption per `generate()` call for models that
        # support it (the Qwen3-VL family), or how many requests to have in
        # flight at once for the remote-endpoint model.
        self.batch_size_spin_box = FocusedScrollSettingsSpinBox(
            key='batch_size', default=1, minimum=1, maximum=64)
        self.attention_implementation_combo_box = (
            FocusedScrollSettingsComboBox(key='attention_implementation'))
        self.attention_implementation_combo_box.addItems(
            list(ATTENTION_IMPLEMENTATIONS.keys()))
        # How many GPUs to caption on at once when the device is
        # `Multiple GPUs (parallel)`: one model instance per GPU, with the
        # selected images split round-robin between them.
        self.parallel_gpu_count_spin_box = FocusedScrollSettingsSpinBox(
            key='parallel_gpu_count', default=2, minimum=2, maximum=8)
        # A plain (not settings-backed) combo box: it only picks which
        # preset to apply to the generation parameter widgets below, which
        # are themselves what gets persisted.
        self.generation_preset_combo_box = QComboBox()
        self.generation_preset_combo_box.addItems(
            list(GENERATION_PRESETS.keys()))
        self.generation_preset_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.generation_preset_combo_box.setMinimumContentsLength(20)
        self.generation_preset_apply_button = QPushButton('Apply Preset')
        self.generation_preset_apply_button.clicked.connect(
            self.apply_generation_preset)
        self.generation_preset_row_container = QWidget()
        generation_preset_row_layout = QVBoxLayout(
            self.generation_preset_row_container)
        generation_preset_row_layout.setContentsMargins(0, 0, 0, 0)
        generation_preset_row_layout.addWidget(
            self.generation_preset_combo_box)
        generation_preset_row_layout.addWidget(
            self.generation_preset_apply_button)
        advanced_settings_form.addRow(bad_forced_words_form)
        advanced_settings_form.addRow(
            label_with_help(
                'Generation preset',
                'Apply a set of generation parameter defaults below, '
                'tuned to a specific captioning model.'),
            self.generation_preset_row_container)
        advanced_settings_form.addRow(HorizontalLine())
        advanced_settings_form.addRow(
            label_with_help(
                'Minimum tokens',
                'The minimum number of new tokens to generate, '
                'including any reasoning/thinking tokens for a model '
                'that uses them.'),
            self.min_new_token_count_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Maximum tokens',
                'The maximum number of new tokens to generate. Too low '
                'can cut a caption off mid-sentence, especially for a '
                'model with a reasoning step.'),
            self.max_new_token_count_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Number of beams',
                'The number of candidate sequences to explore at once '
                '(beam search). Higher can improve quality but is '
                'slower and uses more memory. 1 disables beam search.'),
            self.beam_count_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Length penalty',
                'Favors longer (above 1) or shorter (below 1) captions '
                'when using beam search. Has no effect with only 1 '
                'beam.'),
            self.length_penalty_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Use sampling',
                'Pick each next word randomly, weighted by probability, '
                'instead of always picking the most likely one. Needed '
                'for Temperature/Top-k/Top-p to have any effect.'),
            self.use_sampling_check_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Temperature',
                'How random the sampling is. Lower is more focused, '
                'higher is more random. Only applies when Use sampling '
                'is checked.'),
            self.temperature_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Top-k',
                'Only sample from the k most likely next words. 0 '
                'disables this limit. Only applies when Use sampling '
                'is checked.'),
            self.top_k_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Top-p',
                'Only sample from the smallest set of words whose '
                'combined probability reaches p. 1 disables this '
                'limit. Only applies when Use sampling is checked.'),
            self.top_p_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Repetition penalty',
                'Penalizes words that already appeared in the caption, '
                'to discourage repetition. 1 disables this.'),
            self.repetition_penalty_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'No repeat n-gram size',
                'Blocks the model from repeating any sequence of this '
                'many words. 0 disables this; a model with a reasoning '
                'step sometimes needs it disabled, since it can '
                'legitimately repeat phrases while reasoning.'),
            self.no_repeat_ngram_size_spin_box)
        advanced_settings_form.addRow(HorizontalLine())
        advanced_settings_form.addRow(
            label_with_help(
                'Batch size',
                'The number of images to caption in a single generate() '
                'call. Higher can be faster overall but uses more '
                'VRAM; not every model supports batching.'),
            self.batch_size_spin_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Attention implementation',
                'Which attention implementation the model uses. '
                'Default lets the model/transformers choose '
                'automatically; SDPA and FlashAttention 2 can be '
                'faster or use less memory, depending on the model '
                'and GPU.'),
            self.attention_implementation_combo_box)
        advanced_settings_form.addRow(
            label_with_help(
                'Parallel GPU count',
                'The number of GPUs to split captioning across - one '
                'model instance per GPU, with the image list divided '
                'round-robin between them.'),
            self.parallel_gpu_count_spin_box)
        self.torch_compile_container, self.torch_compile_check_box = (
            self.create_checkbox_row(
                'Compile model (torch.compile, experimental)',
                key='torch_compile', default=False,
                tooltip='Compile the model with torch.compile for '
                       'potentially faster generation after a slow '
                       'initial compilation step. Experimental; not '
                       'compatible with 4-bit/8-bit quantization or '
                       'SageAttention.'))
        advanced_settings_form.addRow(self.torch_compile_container)
        (self.quantize_kv_cache_container,
         self.quantize_kv_cache_check_box) = self.create_checkbox_row(
            'Quantize KV cache (fp8)', key='quantize_kv_cache', default=False,
            tooltip='Store the attention key/value cache in 8-bit '
                   'floating point to use less VRAM on long '
                   'generations, at a small quality/speed tradeoff.')
        advanced_settings_form.addRow(self.quantize_kv_cache_container)
        advanced_settings_form.addRow(HorizontalLine())
        advanced_settings_form.addRow(
            label_with_help(
                'GPU index',
                'Which GPU to use, if more than one is available.'),
            self.gpu_index_spin_box)
        self.advanced_settings_form_container.hide()

        self.addLayout(basic_settings_form)
        self.addWidget(self.wd_tagger_settings_form_container)
        self.addWidget(self.remote_api_settings_form_container)
        self.horizontal_line = HorizontalLine()
        self.addWidget(self.horizontal_line)
        self.addWidget(self.toggle_advanced_settings_form_button)
        self.addWidget(self.advanced_settings_form_container)
        self.addStretch()

        self.model_combo_box.currentTextChanged.connect(
            self.show_settings_for_model)
        self.device_combo_box.currentTextChanged.connect(
            self.set_quantization_options_visibility)
        self.toggle_advanced_settings_form_button.clicked.connect(
            self.toggle_advanced_settings_form)
        # Make sure the minimum new token count is less than or equal to the
        # maximum new token count.
        self.min_new_token_count_spin_box.valueChanged.connect(
            self.max_new_token_count_spin_box.setMinimum)
        self.max_new_token_count_spin_box.valueChanged.connect(
            self.min_new_token_count_spin_box.setMaximum)

        self.show_settings_for_model(self.model_combo_box.currentText())
        self.set_quantization_options_visibility(
            self.device_combo_box.currentText())
        if not self.is_bitsandbytes_available:
            self.load_in_4_bit_check_box.setChecked(False)
            self.load_in_8_bit_check_box.setChecked(False)

        # Tracks which model's settings are currently loaded, so
        # `on_model_changed` below knows which model to save the outgoing
        # settings under. Set before connecting the signal so the initial
        # setup above doesn't trigger a spurious save.
        self.last_model_id = self.model_combo_box.currentText()
        self.model_combo_box.currentTextChanged.connect(
            self.on_model_changed)

        self.reload_profiles()
        # Connect this after the widgets are populated so that loading a
        # profile does not race with the initial setup above.
        self.profile_combo_box.currentTextChanged.connect(self.load_profile)

    def reload_profiles(self):
        self.profile_combo_box.blockSignals(True)
        self.profile_combo_box.clear()
        self.profile_combo_box.addItem('')
        self.profile_combo_box.addItems(list(get_caption_profiles()))
        self.profile_combo_box.blockSignals(False)

    @Slot()
    def save_profile(self):
        name, is_confirmed = QInputDialog.getText(
            None, 'Save Captioning Profile', 'Profile name:')
        name = name.strip()
        if not is_confirmed or not name:
            return
        profiles = get_caption_profiles()
        if name in profiles:
            reply = QMessageBox.question(
                None, 'Save Captioning Profile',
                f'A profile named "{name}" already exists. Overwrite it?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        profiles[name] = self.get_caption_settings()
        save_caption_profiles(profiles)
        self.reload_profiles()
        self.profile_combo_box.blockSignals(True)
        self.profile_combo_box.setCurrentText(name)
        self.profile_combo_box.blockSignals(False)

    @Slot(str)
    def load_profile(self, name: str):
        if not name:
            return
        profiles = get_caption_profiles()
        caption_settings = profiles.get(name)
        if caption_settings is None:
            return
        self.apply_caption_settings(caption_settings)

    @Slot()
    def delete_profile(self):
        name = self.profile_combo_box.currentText()
        if not name:
            return
        profiles = get_caption_profiles()
        if name not in profiles:
            return
        reply = QMessageBox.question(
            None, 'Delete Captioning Profile',
            f'Delete the captioning profile "{name}"?')
        if reply != QMessageBox.StandardButton.Yes:
            return
        del profiles[name]
        save_caption_profiles(profiles)
        self.reload_profiles()

    @Slot(str)
    def on_model_changed(self, new_model_id: str):
        """
        Give each model its own remembered settings (prompt, generation
        parameters, quantization, etc.): save the outgoing model's current
        settings before switching away from it, then restore the
        newly-selected model's own settings if it has any saved.
        """
        old_model_id = self.last_model_id
        if old_model_id and old_model_id != new_model_id:
            settings_to_save = self.get_caption_settings()
            # `get_caption_settings()` reads `model_id` from the combo box,
            # which has already changed to `new_model_id` by the time this
            # slot runs - save it under the model it actually describes.
            settings_to_save['model_id'] = old_model_id
            all_model_settings = get_model_settings()
            all_model_settings[old_model_id] = settings_to_save
            save_model_settings(all_model_settings)
        self.last_model_id = new_model_id
        saved_settings = get_model_settings().get(new_model_id)
        if saved_settings is not None:
            self.apply_caption_settings(saved_settings)

    def apply_caption_settings(self, caption_settings: dict):
        """
        Apply a (possibly older/incomplete) saved settings dict to the form's
        widgets. Every widget is a `Settings*` subclass that persists to
        `QSettings` on change, so this also becomes the new live
        configuration.
        """
        current = self.get_caption_settings()
        generation_parameters = {
            **current['generation_parameters'],
            **caption_settings.get('generation_parameters', {})
        }
        wd_tagger_settings = {
            **current['wd_tagger_settings'],
            **caption_settings.get('wd_tagger_settings', {})
        }
        self.model_combo_box.setCurrentText(
            caption_settings.get('model_id', current['model_id']))
        self.prompt_text_edit.setPlainText(
            caption_settings.get('prompt', current['prompt']))
        self.caption_start_line_edit.setText(
            caption_settings.get('caption_start', current['caption_start']))
        self.caption_position_combo_box.setCurrentText(
            caption_settings.get('caption_position',
                                 current['caption_position']))
        self.skip_already_captioned_check_box.setChecked(
            caption_settings.get('skip_already_captioned',
                                 current['skip_already_captioned']))
        self.test_run_image_count_spin_box.setValue(
            caption_settings.get('test_run_image_count',
                                 current['test_run_image_count']))
        self.device_combo_box.setCurrentText(
            caption_settings.get('device', current['device']))
        self.gpu_index_spin_box.setValue(
            caption_settings.get('gpu_index', current['gpu_index']))
        self.parallel_gpu_count_spin_box.setValue(
            caption_settings.get('parallel_gpu_count',
                                 current['parallel_gpu_count']))
        self.load_in_4_bit_check_box.setChecked(
            caption_settings.get('load_in_4_bit', current['load_in_4_bit']))
        self.load_in_8_bit_check_box.setChecked(
            caption_settings.get('load_in_8_bit', current['load_in_8_bit']))
        self.cpu_offload_check_box.setChecked(
            caption_settings.get('cpu_offload', current['cpu_offload']))
        self.sage_attention_check_box.setChecked(
            caption_settings.get('sage_attention', current['sage_attention']))
        self.disable_thinking_check_box.setChecked(
            caption_settings.get('disable_thinking',
                                 current['disable_thinking']))
        self.min_image_megapixels_spin_box.setValue(
            caption_settings.get('min_image_megapixels',
                                 current['min_image_megapixels']))
        self.max_image_megapixels_spin_box.setValue(
            caption_settings.get('max_image_megapixels',
                                 current['max_image_megapixels']))
        self.api_base_url_line_edit.setText(
            caption_settings.get('api_base_url', current['api_base_url']))
        self.api_key_line_edit.setText(
            caption_settings.get('api_key', current['api_key']))
        self.api_model_name_line_edit.setText(
            caption_settings.get('api_model_name',
                                 current['api_model_name']))
        self.batch_size_spin_box.setValue(
            caption_settings.get('batch_size', current['batch_size']))
        self.attention_implementation_combo_box.setCurrentText(
            caption_settings.get('attention_implementation',
                                 current['attention_implementation']))
        self.torch_compile_check_box.setChecked(
            caption_settings.get('torch_compile', current['torch_compile']))
        self.quantize_kv_cache_check_box.setChecked(
            caption_settings.get('quantize_kv_cache',
                                 current['quantize_kv_cache']))
        self.wd_tagger_load_in_8_bit_check_box.setChecked(
            caption_settings.get('wd_tagger_load_in_8_bit',
                                 current['wd_tagger_load_in_8_bit']))
        self.remove_tag_separators_check_box.setChecked(
            caption_settings.get('remove_tag_separators',
                                 current['remove_tag_separators']))
        self.bad_words_line_edit.setText(
            caption_settings.get('bad_words', current['bad_words']))
        self.forced_words_line_edit.setText(
            caption_settings.get('forced_words', current['forced_words']))
        self.min_new_token_count_spin_box.setValue(
            generation_parameters['min_new_tokens'])
        self.max_new_token_count_spin_box.setValue(
            generation_parameters['max_new_tokens'])
        self.beam_count_spin_box.setValue(generation_parameters['num_beams'])
        self.length_penalty_spin_box.setValue(
            generation_parameters['length_penalty'])
        self.use_sampling_check_box.setChecked(
            generation_parameters['do_sample'])
        self.temperature_spin_box.setValue(
            generation_parameters['temperature'])
        self.top_k_spin_box.setValue(generation_parameters['top_k'])
        self.top_p_spin_box.setValue(generation_parameters['top_p'])
        self.repetition_penalty_spin_box.setValue(
            generation_parameters['repetition_penalty'])
        self.no_repeat_ngram_size_spin_box.setValue(
            generation_parameters['no_repeat_ngram_size'])
        self.show_probabilities_check_box.setChecked(
            wd_tagger_settings['show_probabilities'])
        self.min_probability_spin_box.setValue(
            wd_tagger_settings['min_probability'])
        self.max_tags_spin_box.setValue(wd_tagger_settings['max_tags'])
        self.tags_to_exclude_text_edit.setPlainText(
            wd_tagger_settings['tags_to_exclude'])

    @staticmethod
    def create_checkbox_row(
            label_text: str, key: str, default: bool,
            tooltip: str | None = None
    ) -> tuple[QWidget, SettingsBigCheckBox]:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        check_box = SettingsBigCheckBox(key=key, default=default)
        layout.addWidget(QLabel(label_text))
        if tooltip:
            layout.addWidget(create_help_label(tooltip))
        layout.addWidget(check_box)
        return container, check_box

    @Slot()
    def update_pixel_budget_label(self):
        pixels_per_megapixel = 1_000_000
        max_pixels = (self.max_image_megapixels_spin_box.value()
                      * pixels_per_megapixel)
        visual_tokens = round(max_pixels / PIXELS_PER_VISUAL_TOKEN)
        self.pixel_budget_label.setText(
            f'Up to about {visual_tokens} visual tokens per image')

    @Slot()
    def insert_prompt_template(self):
        model = self.prompt_template_model_combo_box.currentText()
        purpose = self.prompt_template_purpose_combo_box.currentText()
        captioner = self.prompt_template_captioner_combo_box.currentText()
        template = PROMPT_TEMPLATES[model][purpose]
        captioner_suffix = CAPTIONER_PROMPT_SUFFIXES.get(captioner)
        if captioner_suffix:
            template = f'{template} {captioner_suffix}'
        self.prompt_text_edit.setPlainText(template)

    @Slot()
    def apply_generation_preset(self):
        captioner = self.generation_preset_combo_box.currentText()
        preset = GENERATION_PRESETS[captioner]
        self.min_new_token_count_spin_box.setValue(preset['min_new_tokens'])
        self.max_new_token_count_spin_box.setValue(preset['max_new_tokens'])
        self.beam_count_spin_box.setValue(preset['num_beams'])
        self.length_penalty_spin_box.setValue(preset['length_penalty'])
        self.use_sampling_check_box.setChecked(preset['do_sample'])
        self.temperature_spin_box.setValue(preset['temperature'])
        self.top_k_spin_box.setValue(preset['top_k'])
        self.top_p_spin_box.setValue(preset['top_p'])
        self.repetition_penalty_spin_box.setValue(
            preset['repetition_penalty'])
        self.no_repeat_ngram_size_spin_box.setValue(
            preset['no_repeat_ngram_size'])

    @Slot()
    def apply_load_preset(self):
        preset = LOAD_PRESETS[self.load_preset_combo_box.currentText()]
        if 'load_in_4_bit' in preset:
            self.load_in_4_bit_check_box.setChecked(preset['load_in_4_bit'])
        if 'load_in_8_bit' in preset:
            self.load_in_8_bit_check_box.setChecked(preset['load_in_8_bit'])
        if 'cpu_offload' in preset:
            self.cpu_offload_check_box.setChecked(preset['cpu_offload'])
        if 'sage_attention' in preset:
            self.sage_attention_check_box.setChecked(preset['sage_attention'])
        if 'disable_thinking' in preset:
            self.disable_thinking_check_box.setChecked(
                preset['disable_thinking'])
        if 'wd_tagger_load_in_8_bit' in preset:
            self.wd_tagger_load_in_8_bit_check_box.setChecked(
                preset['wd_tagger_load_in_8_bit'])
        if 'attention_implementation' in preset:
            self.attention_implementation_combo_box.setCurrentText(
                preset['attention_implementation'])
        if 'torch_compile' in preset:
            self.torch_compile_check_box.setChecked(preset['torch_compile'])
        if 'quantize_kv_cache' in preset:
            self.quantize_kv_cache_check_box.setChecked(
                preset['quantize_kv_cache'])
        if 'batch_size' in preset:
            self.batch_size_spin_box.setValue(preset['batch_size'])
        if 'min_image_megapixels' in preset:
            self.min_image_megapixels_spin_box.setValue(
                preset['min_image_megapixels'])
        if 'max_image_megapixels' in preset:
            self.max_image_megapixels_spin_box.setValue(
                preset['max_image_megapixels'])

    @Slot()
    def show_model_manager(self):
        model_manager_dialog = ModelManagerDialog(parent=None)
        model_manager_dialog.exec()
        current_model_id = self.model_combo_box.currentText()
        self.model_combo_box.clear()
        self.model_combo_box.addItems(self.get_local_model_paths())
        self.model_combo_box.addItems(get_all_models())
        self.model_combo_box.setCurrentText(current_model_id)

    def get_local_model_paths(self) -> list[str]:
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        if not models_directory_path:
            return []
        models_directory_path = Path(models_directory_path)
        print(f'Loading local auto-captioning model paths under '
              f'{models_directory_path}...')
        # Auto-captioning models have a `config.json` file.
        config_paths = set(models_directory_path.glob('**/config.json'))
        # WD Tagger models have a `selected_tags.csv` file.
        selected_tags_paths = set(
            models_directory_path.glob('**/selected_tags.csv'))
        model_directory_paths = [str(path.parent) for path
                                 in config_paths | selected_tags_paths]
        model_directory_paths.sort()
        print(f'Loaded {len(model_directory_paths)} model '
              f'{pluralize("path", len(model_directory_paths))}.')
        return model_directory_paths

    @Slot(str)
    def show_settings_for_model(self, model_id: str):
        is_wd_tagger_model = get_model_class(model_id) == WdTagger
        is_remote_api_model = is_remote_endpoint_model(model_id)
        is_qwen3_vl_family_model = issubclass(get_model_class(model_id),
                                              Qwen3VL)
        # (visible, widget) per widget, computed for the selected model.
        visibility = [
            (is_wd_tagger_model, self.wd_tagger_settings_form_container),
            (is_remote_api_model, self.remote_api_settings_form_container),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.prompt_label),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.prompt_text_edit),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.caption_start_label),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.caption_start_line_edit),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.device_label),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.device_combo_box),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.load_in_4_bit_container),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.load_in_8_bit_container),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.cpu_offload_container),
            (not is_wd_tagger_model and not is_remote_api_model,
             self.sage_attention_container),
            # Only the Qwen3-VL family's chat template reads
            # `disable_thinking` (see `Qwen3VL.format_prompt()`); hide the
            # checkbox for every other model instead of leaving a setting on
            # screen that does nothing.
            (not is_wd_tagger_model and is_qwen3_vl_family_model,
             self.disable_thinking_container),
            # The pixel budget is applied per call by the Qwen3-VL family
            # (`Qwen3VL.get_size_kwargs()`); other models have their own
            # fixed preprocessing.
            (not is_wd_tagger_model and is_qwen3_vl_family_model,
             self.pixel_budget_settings_form_container),
            (not is_wd_tagger_model, self.remove_tag_separators_container),
            (not is_wd_tagger_model, self.compare_prompts_button),
            (not is_wd_tagger_model,
             self.skip_already_captioned_container),
            (not is_wd_tagger_model, self.test_run_image_count_spin_box),
            (not is_wd_tagger_model, self.load_preset_row_container),
            (not is_wd_tagger_model, self.horizontal_line),
            (not is_wd_tagger_model,
             self.toggle_advanced_settings_form_button),
            (not is_wd_tagger_model, self.advanced_settings_form_container)
        ]
        for is_visible, widget in visibility:
            widget.setVisible(is_visible)
        test_run_label = (self.basic_settings_form.labelForField(
            self.test_run_image_count_spin_box))
        test_run_label.setVisible(not is_wd_tagger_model)
        self.set_quantization_options_visibility(
            self.device_combo_box.currentText())

    @Slot(str)
    def set_quantization_options_visibility(self, device: str):
        model_id = self.model_combo_box.currentText()
        is_wd_tagger_model = get_model_class(model_id) == WdTagger
        is_remote_api_model = is_remote_endpoint_model(model_id)
        is_gpu_device = device in (CaptionDevice.GPU,
                                   CaptionDevice.GPUS_PARALLEL)
        if is_wd_tagger_model or is_remote_api_model:
            self.load_in_4_bit_container.setVisible(False)
            self.load_in_8_bit_container.setVisible(False)
            self.cpu_offload_container.setVisible(False)
            self.sage_attention_container.setVisible(False)
            self.torch_compile_container.setVisible(False)
            self.quantize_kv_cache_container.setVisible(False)
            self.attention_implementation_combo_box.setVisible(False)
            attention_label = (self.advanced_settings_form.labelForField(
                self.attention_implementation_combo_box))
            attention_label.setVisible(False)
            return
        is_load_in_4_bit_available = (self.is_bitsandbytes_available
                                      and is_gpu_device)
        self.load_in_4_bit_container.setVisible(is_load_in_4_bit_available)
        self.load_in_8_bit_container.setVisible(is_load_in_4_bit_available)
        self.cpu_offload_container.setVisible(is_gpu_device)
        self.sage_attention_container.setVisible(is_gpu_device)
        self.torch_compile_container.setVisible(is_gpu_device)
        self.quantize_kv_cache_container.setVisible(is_gpu_device)
        self.attention_implementation_combo_box.setVisible(is_gpu_device)
        attention_label = (self.advanced_settings_form.labelForField(
            self.attention_implementation_combo_box))
        attention_label.setVisible(is_gpu_device)
        is_parallel_device = device == CaptionDevice.GPUS_PARALLEL
        self.parallel_gpu_count_spin_box.setVisible(is_parallel_device)
        parallel_gpu_label = (self.advanced_settings_form.labelForField(
            self.parallel_gpu_count_spin_box))
        parallel_gpu_label.setVisible(is_parallel_device)

    @Slot()
    def toggle_advanced_settings_form(self):
        if self.advanced_settings_form_container.isHidden():
            self.advanced_settings_form_container.show()
            self.toggle_advanced_settings_form_button.setText(
                'Hide Advanced Settings')
        else:
            self.advanced_settings_form_container.hide()
            self.toggle_advanced_settings_form_button.setText(
                'Show Advanced Settings')

    def get_caption_settings(self) -> dict:
        return {
            'model_id': self.model_combo_box.currentText(),
            'prompt': self.prompt_text_edit.toPlainText(),
            'caption_start': self.caption_start_line_edit.text(),
            'caption_position': self.caption_position_combo_box.currentText(),
            'skip_already_captioned':
                self.skip_already_captioned_check_box.isChecked(),
            'test_run_image_count':
                self.test_run_image_count_spin_box.value(),
            'device': self.device_combo_box.currentText(),
            'gpu_index': self.gpu_index_spin_box.value(),
            'parallel_gpu_count': self.parallel_gpu_count_spin_box.value(),
            'load_in_4_bit': self.load_in_4_bit_check_box.isChecked(),
            'load_in_8_bit': self.load_in_8_bit_check_box.isChecked(),
            'cpu_offload': self.cpu_offload_check_box.isChecked(),
            'sage_attention': self.sage_attention_check_box.isChecked(),
            'disable_thinking': self.disable_thinking_check_box.isChecked(),
            'min_image_megapixels':
                self.min_image_megapixels_spin_box.value(),
            'max_image_megapixels':
                self.max_image_megapixels_spin_box.value(),
            'api_base_url': self.api_base_url_line_edit.text(),
            'api_key': self.api_key_line_edit.text(),
            'api_model_name': self.api_model_name_line_edit.text(),
            'batch_size': self.batch_size_spin_box.value(),
            'attention_implementation':
                self.attention_implementation_combo_box.currentText(),
            'torch_compile': self.torch_compile_check_box.isChecked(),
            'quantize_kv_cache':
                self.quantize_kv_cache_check_box.isChecked(),
            'wd_tagger_load_in_8_bit':
                self.wd_tagger_load_in_8_bit_check_box.isChecked(),
            'remove_tag_separators':
                self.remove_tag_separators_check_box.isChecked(),
            'bad_words': self.bad_words_line_edit.text(),
            'forced_words': self.forced_words_line_edit.text(),
            'generation_parameters': {
                'min_new_tokens': self.min_new_token_count_spin_box.value(),
                'max_new_tokens': self.max_new_token_count_spin_box.value(),
                'num_beams': self.beam_count_spin_box.value(),
                'length_penalty': self.length_penalty_spin_box.value(),
                'do_sample': self.use_sampling_check_box.isChecked(),
                'temperature': self.temperature_spin_box.value(),
                'top_k': self.top_k_spin_box.value(),
                'top_p': self.top_p_spin_box.value(),
                'repetition_penalty': self.repetition_penalty_spin_box.value(),
                'no_repeat_ngram_size':
                    self.no_repeat_ngram_size_spin_box.value()
            },
            'wd_tagger_settings': {
                'show_probabilities':
                    self.show_probabilities_check_box.isChecked(),
                'min_probability': self.min_probability_spin_box.value(),
                'max_tags': self.max_tags_spin_box.value(),
                'tags_to_exclude':
                    self.tags_to_exclude_text_edit.toPlainText()
            }
        }


@Slot()
def restore_stdout_and_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class AutoCaptioner(QDockWidget):
    caption_generated = Signal(QModelIndex, str, list)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.settings = get_settings()
        self.is_captioning = False
        self.captioning_thread = None
        self.processor = None
        self.model = None
        self.model_id: str | None = None
        self.model_device_type: str | None = None
        self.is_model_loaded_in_4_bit = None
        self.is_model_loaded_in_8_bit = None
        self.is_model_cpu_offloaded = None
        self.is_model_using_sage_attention = None
        self.is_model_torch_compiled = None
        self.is_model_attention_implementation = None
        self.is_wd_tagger_loaded_in_8_bit = None
        # Whether the last block of text in the console text edit should be
        # replaced with the next block of text that is outputted.
        self.replace_last_console_text_edit_block = False

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('auto_captioner')
        self.setWindowTitle('Auto-Captioner')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.start_cancel_button = TallPushButton('Start Auto-Captioning')
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat('%v / %m images captioned (%p%)')
        self.progress_bar.hide()
        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 4)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.start_cancel_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.console_text_edit)
        self.caption_settings_form = CaptionSettingsForm()
        layout.addLayout(self.caption_settings_form)
        # The unload/compare buttons live in the form (for layout), but
        # their actions use this widget's captioning/model-cache state.
        self.caption_settings_form.unload_model_button.clicked.connect(
            self.unload_model)
        self.caption_settings_form.compare_prompts_button.clicked.connect(
            self.compare_prompts)
        self.caption_settings_form.model_chains_button.clicked.connect(
            self.show_model_chains_dialog)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.start_cancel_button.clicked.connect(
            self.start_or_cancel_captioning)

    @Slot()
    def unload_model(self):
        if self.is_captioning or self.model is None:
            return
        # Drop the widget-level model cache (the single-GPU path); a
        # parallel multi-GPU run's models live in per-device holders that
        # are freed when their run finishes.
        self.processor = None
        self.model = None
        self.model_id = None
        self.model_device_type = None
        self.is_model_loaded_in_4_bit = None
        self.is_model_loaded_in_8_bit = None
        self.is_model_cpu_offloaded = None
        self.is_model_using_sage_attention = None
        self.is_model_torch_compiled = None
        self.is_model_attention_implementation = None
        self.is_wd_tagger_loaded_in_8_bit = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.caption_settings_form.unload_model_button.setEnabled(False)
        QMessageBox.information(
            None, 'Model Unloaded',
            'The loaded model was removed from memory and its VRAM was '
            'freed. It will be loaded again the next time you caption.')

    @Slot()
    def compare_prompts(self):
        if self.is_captioning:
            QMessageBox.warning(
                self, 'Auto-Captioner Busy',
                'Captioning is already running. Wait for it to finish '
                'before comparing prompts.')
            return
        dialog = PromptComparisonDialog(self)
        dialog.exec()

    @Slot()
    def show_model_chains_dialog(self):
        if self.is_captioning:
            QMessageBox.warning(
                self, 'Auto-Captioner Busy',
                'Captioning is already running. Wait for it to finish '
                'before running a model chain.')
            return
        dialog = ModelChainDialog(self)
        dialog.exec()

    @Slot(int)
    def show_pilot_run_prompt(self, image_count: int):
        """
        Test-run prompt: shown (non-modal, so the user can review the
        captions in the main window first) after the first N images of a
        run; the thread pauses until a button is pressed.
        """
        message_box = QMessageBox(self)
        message_box.setWindowTitle('Test Run Finished')
        message_box.setText(
            f'{image_count} image(s) captioned. The captions are already '
            f'applied to their tags - review them in the image list, then '
            f'choose whether to continue with the remaining images.')
        continue_button = message_box.addButton(
            'Continue captioning', QMessageBox.ButtonRole.YesRole)
        stop_button = message_box.addButton(
            'Stop here', QMessageBox.ButtonRole.NoRole)
        message_box.setWindowModality(Qt.WindowModality.NonModal)

        def finish_pilot_run():
            clicked_button = message_box.clickedButton()
            self.captioning_thread.continue_after_pilot = (
                clicked_button is continue_button)
            message_box.deleteLater()

        message_box.finished.connect(finish_pilot_run)
        self.pilot_message_box = message_box
        message_box.show()
        message_box.raise_()

    @Slot()
    def offer_failed_image_selection(self):
        """
        After a run with failures, offer to select the failed images in the
        image list, so retrying them is just pressing Start again.
        """
        thread = self.captioning_thread
        failed_image_indices = getattr(thread, 'failed_image_indices', [])
        if not failed_image_indices:
            return
        failed_count = len(failed_image_indices)
        reply = QMessageBox.question(
            None, 'Captioning Failures',
            f'{failed_count} {pluralize("image", failed_count)} failed '
            f'during captioning (see the Auto-Captioner console for the '
            f'errors). Select them in the image list so you can retry '
            f'just them?')
        if reply == QMessageBox.StandardButton.Yes:
            self.image_list.select_image_indices(failed_image_indices)

    @Slot()
    def start_or_cancel_captioning(self):
        if self.is_captioning:
            # Cancel captioning.
            self.captioning_thread.is_canceled = True
            self.start_cancel_button.setEnabled(False)
            self.start_cancel_button.setText('Canceling Auto-Captioning...')
        else:
            # Start captioning.
            self.generate_captions()

    def set_is_captioning(self, is_captioning: bool):
        self.is_captioning = is_captioning
        button_text = ('Cancel Auto-Captioning' if is_captioning
                       else 'Start Auto-Captioning')
        self.start_cancel_button.setText(button_text)

    @Slot(str)
    def update_console_text_edit(self, text: str):
        # '\x1b[A' is the ANSI escape sequence for moving the cursor up.
        if text == '\x1b[A':
            self.replace_last_console_text_edit_block = True
            return
        text = text.strip()
        if not text:
            return
        if self.console_text_edit.isHidden():
            self.console_text_edit.show()
        if self.replace_last_console_text_edit_block:
            self.replace_last_console_text_edit_block = False
            # Select and remove the last block of text.
            self.console_text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.console_text_edit.moveCursor(
                QTextCursor.MoveOperation.StartOfBlock,
                QTextCursor.MoveMode.KeepAnchor)
            self.console_text_edit.textCursor().removeSelectedText()
            # Delete the newline.
            self.console_text_edit.textCursor().deletePreviousChar()
        self.console_text_edit.appendPlainText(text)

    @Slot()
    def show_alert(self):
        if self.captioning_thread.is_canceled:
            return
        if self.captioning_thread.is_error:
            icon = QMessageBox.Icon.Critical
            text = ('An error occurred during captioning. See the '
                    'Auto-Captioner console for more information.')
        else:
            icon = QMessageBox.Icon.Information
            text = 'Captioning has finished.'
        alert = QMessageBox()
        alert.setIcon(icon)
        alert.setText(text)
        alert.exec()

    @Slot()
    def generate_captions(self):
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        show_alert_when_finished = False
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
            show_alert_when_finished = (confirmation_dialog
                                        .show_alert_check_box.isChecked())
        self.set_is_captioning(True)
        self.caption_settings_form.unload_model_button.setEnabled(False)
        caption_settings = self.caption_settings_form.get_caption_settings()
        if caption_settings['caption_position'] != CaptionPosition.DO_NOT_ADD:
            self.image_list_model.add_to_undo_stack(
                action_name=f'Generate '
                            f'{pluralize("Caption", selected_image_count)}',
                should_ask_for_confirmation=selected_image_count > 1)
        if selected_image_count > 1:
            self.progress_bar.setRange(0, selected_image_count)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        tag_separator = get_tag_separator()
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        models_directory_path = (Path(models_directory_path)
                                 if models_directory_path else None)
        self.captioning_thread = CaptioningThread(
            self, self.image_list_model, selected_image_indices,
            caption_settings, tag_separator, models_directory_path)
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            self.caption_generated)
        self.captioning_thread.progress_bar_update_requested.connect(
            self.progress_bar.setValue)
        self.captioning_thread.progress_bar_range_update_requested.connect(
            self.progress_bar.setMaximum)
        self.captioning_thread.progress_bar_format_update_requested.connect(
            self.progress_bar.setFormat)
        self.captioning_thread.pilot_run_finished_requested.connect(
            self.show_pilot_run_prompt)
        self.captioning_thread.finished.connect(
            self.offer_failed_image_selection)
        self.captioning_thread.finished.connect(
            lambda: self.caption_settings_form.unload_model_button
            .setEnabled(self.model is not None))
        self.captioning_thread.finished.connect(
            lambda: self.set_is_captioning(False))
        self.captioning_thread.finished.connect(restore_stdout_and_stderr)
        self.captioning_thread.finished.connect(self.progress_bar.hide)
        self.captioning_thread.finished.connect(
            lambda: self.start_cancel_button.setEnabled(True))
        if show_alert_when_finished:
            self.captioning_thread.finished.connect(self.show_alert)
        # Redirect `stdout` and `stderr` so that the outputs are displayed in
        # the console text edit.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()
