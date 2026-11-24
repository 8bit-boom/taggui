from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QWidget

from auto_captioning.generation_presets import GENERATION_PRESETS
from auto_captioning.models.qwen3_vl import Qwen3VL
from auto_captioning.prompt_templates import (CAPTIONER_PROMPT_SUFFIXES,
                                              PROMPT_TEMPLATE_CAPTIONERS,
                                              PROMPT_TEMPLATE_PURPOSES,
                                              PROMPT_TEMPLATES)
from widgets.auto_captioner import CaptionSettingsForm


@pytest.fixture
def caption_settings_form(qtbot, isolated_settings):
    widget = QWidget()
    form = CaptionSettingsForm()
    widget.setLayout(form)
    qtbot.addWidget(widget)
    yield form


def test_prompt_template_data_is_complete():
    assert PROMPT_TEMPLATES
    assert PROMPT_TEMPLATE_PURPOSES
    assert len(PROMPT_TEMPLATE_PURPOSES) == len(set(PROMPT_TEMPLATE_PURPOSES))
    for templates in PROMPT_TEMPLATES.values():
        assert set(templates) == set(PROMPT_TEMPLATE_PURPOSES)
        assert all(isinstance(text, str) and text.strip()
                   for text in templates.values())

    assert len(PROMPT_TEMPLATE_CAPTIONERS) == len(set(PROMPT_TEMPLATE_CAPTIONERS))
    assert '(none)' in PROMPT_TEMPLATE_CAPTIONERS
    assert set(CAPTIONER_PROMPT_SUFFIXES) == (
        set(PROMPT_TEMPLATE_CAPTIONERS) - {'(none)'})
    assert all(isinstance(text, str) and text.strip()
               for text in CAPTIONER_PROMPT_SUFFIXES.values())


@pytest.mark.parametrize(
    'model,purpose,captioner',
    [('SDXL', 'General', '(none)'),
     ('Flux.1', 'Character', 'Qwen3-VL'),
     ('Anima', 'Style', 'Qwen3.5'),
     ('SDXL', 'Landscape', 'JoyCaption')])
def test_insert_prompt_template(caption_settings_form, model, purpose,
                                captioner):
    form = caption_settings_form
    for combo, selection in (
            (form.prompt_template_model_combo_box, model),
            (form.prompt_template_purpose_combo_box, purpose),
            (form.prompt_template_captioner_combo_box, captioner)):
        assert combo.findText(selection) >= 0
        combo.setCurrentText(selection)
    form.prompt_text_edit.setPlainText('Existing prompt')
    form.prompt_template_insert_button.click()

    expected = PROMPT_TEMPLATES[model][purpose]
    if captioner != '(none)':
        expected += ' ' + CAPTIONER_PROMPT_SUFFIXES[captioner]
    assert form.prompt_text_edit.toPlainText() == expected


def test_generation_preset_coverage_and_schema(caption_settings_form):
    form = caption_settings_form
    parameter_keys = set(form.get_caption_settings()['generation_parameters'])
    assert set(GENERATION_PRESETS) == (
        set(PROMPT_TEMPLATE_CAPTIONERS) - {'(none)'})
    for preset in GENERATION_PRESETS.values():
        assert set(preset) == parameter_keys
        for key in ('min_new_tokens', 'max_new_tokens', 'num_beams', 'top_k',
                    'no_repeat_ngram_size'):
            assert type(preset[key]) is int
        assert type(preset['do_sample']) is bool
        assert 1 <= preset['min_new_tokens'] <= preset['max_new_tokens']
        assert preset['num_beams'] >= 1
        assert preset['temperature'] > 0
        assert preset['top_k'] >= 0
        assert 0 < preset['top_p'] <= 1
        assert preset['repetition_penalty'] >= 1
        assert preset['no_repeat_ngram_size'] >= 0
    assert GENERATION_PRESETS['Qwen3-VL'] == GENERATION_PRESETS['Qwen3.5']


@pytest.mark.parametrize('captioner', list(GENERATION_PRESETS))
def test_apply_generation_preset(caption_settings_form, captioner):
    form = caption_settings_form
    preset = GENERATION_PRESETS[captioner]
    widgets = {
        'min_new_tokens': form.min_new_token_count_spin_box,
        'max_new_tokens': form.max_new_token_count_spin_box,
        'num_beams': form.beam_count_spin_box,
        'length_penalty': form.length_penalty_spin_box,
        'temperature': form.temperature_spin_box,
        'top_k': form.top_k_spin_box,
        'top_p': form.top_p_spin_box,
        'repetition_penalty': form.repetition_penalty_spin_box,
        'no_repeat_ngram_size': form.no_repeat_ngram_size_spin_box,
    }
    for key, widget in widgets.items():
        widget.setValue(widget.minimum() if preset[key] != widget.minimum()
                        else widget.maximum())
    form.use_sampling_check_box.setChecked(not preset['do_sample'])
    before = form.get_caption_settings()['generation_parameters']
    assert all(before[key] != preset[key] for key in preset)

    assert form.generation_preset_combo_box.findText(captioner) >= 0
    form.generation_preset_combo_box.setCurrentText(captioner)
    form.generation_preset_apply_button.click()
    assert form.get_caption_settings()['generation_parameters'] == preset


@pytest.mark.parametrize('disable_thinking', [False, True])
def test_qwen_format_prompt(disable_thinking):
    model = Qwen3VL.__new__(Qwen3VL)
    model.disable_thinking = disable_thinking
    model.processor = Mock()
    model.processor.apply_chat_template.return_value = 'formatted prompt'

    assert model.format_prompt('Describe this image.') == 'formatted prompt'
    model.processor.apply_chat_template.assert_called_once_with(
        [{'role': 'user',
          'content': [{'type': 'image'},
                      {'type': 'text', 'text': 'Describe this image.'}]}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=not disable_thinking)
