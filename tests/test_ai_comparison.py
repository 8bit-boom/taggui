"""
Tests for the one-button AI comparison: candidate building, judge-prompt
construction, defensive verdict parsing, and the dialog's end-to-end
sweep -> judge -> apply-winner flow (with a fake caption model and judge).
"""

import tempfile
from pathlib import Path

import pytest

from auto_captioning.ai_comparison import (MAX_CANDIDATES, build_candidates,
                                           build_judge_prompt,
                                           candidate_label,
                                           detect_unsloth_endpoint,
                                           parse_judge_response,
                                           parse_unsloth_llama_command)


# ---------------------------------------------------------------------------
# candidate_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('index,expected', [
    (0, 'A'), (1, 'B'), (25, 'Z'), (26, 'AA'), (27, 'AB'), (51, 'AZ'),
    (52, 'BA'),
])
def test_candidate_label(index, expected):
    assert candidate_label(index) == expected


# ---------------------------------------------------------------------------
# build_candidates
# ---------------------------------------------------------------------------

def _base_settings():
    return {'model_id': 'm', 'device': 'GPU if available',
            'caption_position': 'Insert after last tag',
            'generation_parameters': {'max_new_tokens': 100}}


def test_build_candidates_cartesian_product():
    prompts = ['prompt one', 'prompt two']
    settings = [{'name': 'Current', 'generation_parameters': {'a': 1}},
                {'name': 'Preset', 'generation_parameters': {'a': 2}}]
    candidates = build_candidates(prompts, settings, _base_settings())
    assert len(candidates) == 4
    assert [c['label'] for c in candidates] == ['A', 'B', 'C', 'D']
    # Each candidate forces DO_NOT_ADD and carries its own prompt/params.
    for candidate in candidates:
        assert candidate['settings']['caption_position'] == 'Do not add to tags'
        assert candidate['settings']['skip_already_captioned'] is False
    assert candidates[0]['prompt'] == 'prompt one'
    assert candidates[0]['settings']['generation_parameters'] == {'a': 1}
    assert candidates[1]['settings']['generation_parameters'] == {'a': 2}
    assert candidates[2]['prompt'] == 'prompt two'


def test_build_candidates_skips_blank_prompts():
    candidates = build_candidates(['', '   ', 'real'], [], _base_settings())
    assert len(candidates) == 1
    assert candidates[0]['prompt'] == 'real'


def test_build_candidates_empty_settings_uses_current():
    base = _base_settings()
    candidates = build_candidates(['p'], [], base)
    assert len(candidates) == 1
    assert candidates[0]['settings_name'] == 'Current settings'
    assert candidates[0]['settings']['generation_parameters'] == \
        base['generation_parameters']


def test_build_candidates_no_prompts_is_empty():
    assert build_candidates([], [], _base_settings()) == []
    assert build_candidates(['  '], [], _base_settings()) == []


def test_build_candidates_respects_cap():
    prompts = [f'p{i}' for i in range(10)]
    settings = [{'name': f's{i}', 'generation_parameters': {}}
                for i in range(10)]
    candidates = build_candidates(prompts, settings, _base_settings())
    assert len(candidates) == MAX_CANDIDATES


def test_build_candidates_does_not_mutate_base():
    base = _base_settings()
    build_candidates(['p'], [{'name': 'x', 'generation_parameters': {'z': 9}}],
                     base)
    assert base['caption_position'] == 'Insert after last tag'
    assert base['generation_parameters'] == {'max_new_tokens': 100}


# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------

def test_judge_prompt_includes_labels_captions_and_json_instruction():
    candidates = build_candidates(['p1', 'p2'], [], _base_settings())
    results = [{'image': 'cat.png',
                'captions': {'A': 'a cat', 'B': 'a feline'}}]
    prompt = build_judge_prompt(candidates, results, 'SDXL training')
    assert 'A:' in prompt and 'B:' in prompt
    assert 'cat.png' in prompt
    assert 'a cat' in prompt and 'a feline' in prompt
    assert 'SDXL training' in prompt
    assert 'JSON' in prompt
    assert '"ranking"' in prompt


def test_judge_prompt_marks_empty_captions():
    candidates = build_candidates(['p1', 'p2'], [], _base_settings())
    results = [{'image': 'x.png', 'captions': {'A': 'text', 'B': ''}}]
    prompt = build_judge_prompt(candidates, results)
    assert '(empty)' in prompt


# ---------------------------------------------------------------------------
# parse_judge_response
# ---------------------------------------------------------------------------

LABELS = ['A', 'B', 'C']


def test_parse_valid_response():
    text = ('{"overall": {"scores": {"A": 9, "B": 5, "C": 7}, '
            '"ranking": ["A", "C", "B"], "best": "A", '
            '"reasoning": "A is most detailed."}, "per_image": []}')
    verdict = parse_judge_response(text, LABELS)
    assert verdict['valid'] is True
    assert verdict['best'] == 'A'
    assert verdict['ranking'] == ['A', 'C', 'B']
    assert verdict['scores']['A'] == 9.0
    assert verdict['reasoning'] == 'A is most detailed.'


def test_parse_fenced_json():
    text = ('Here you go:\n```json\n{"overall": {"best": "B", '
            '"ranking": ["B", "A", "C"], "scores": {"B": 8}}}\n```')
    verdict = parse_judge_response(text, LABELS)
    assert verdict['valid'] is True
    assert verdict['best'] == 'B'


def test_parse_tolerates_trailing_commentary():
    text = ('{"overall": {"best": "C", "ranking": ["C", "B", "A"]}} '
            'Hope that helps!')
    verdict = parse_judge_response(text, LABELS)
    assert verdict['best'] == 'C'


def test_parse_appends_omitted_labels():
    text = '{"overall": {"ranking": ["B"], "best": "B"}}'
    verdict = parse_judge_response(text, LABELS)
    assert verdict['ranking'][0] == 'B'
    assert set(verdict['ranking']) == set(LABELS)


def test_parse_invalid_best_falls_back_to_ranking_first():
    text = '{"overall": {"ranking": ["C", "A", "B"], "best": "ZZZ"}}'
    verdict = parse_judge_response(text, LABELS)
    assert verdict['best'] == 'C'


def test_parse_malformed_returns_fallback():
    verdict = parse_judge_response('not json at all', LABELS)
    assert verdict['valid'] is False
    assert verdict['ranking'] == LABELS
    assert verdict['best'] == 'A'


def test_parse_empty_response():
    verdict = parse_judge_response('', LABELS)
    assert verdict['valid'] is False


def test_parse_scores_default_zero_for_missing():
    text = '{"overall": {"scores": {"A": 7}, "ranking": ["A", "B", "C"]}}'
    verdict = parse_judge_response(text, LABELS)
    assert verdict['scores'] == {'A': 7.0, 'B': 0.0, 'C': 0.0}


def test_parse_strips_think_block():
    text = ('<think>let me reason</think>{"overall": {"best": "A", '
            '"ranking": ["A", "B", "C"]}}')
    verdict = parse_judge_response(text, LABELS)
    assert verdict['best'] == 'A'


# ---------------------------------------------------------------------------
# Unsloth Studio auto-detection
# ---------------------------------------------------------------------------

_UNSLOTH_LINE = (
    '[123][stdout] {"event": "Starting llama-server: C:\\\\llama-server.exe '
    '-m C:\\\\models\\\\qwen3.8-27b.gguf --port 62541 --parallel 4 '
    '--flash-attn on --alias DavidAU/Qwen3.8-27B-GGUF -ngl -1 '
    '--api-key-file C:\\\\Users\\\\me\\\\.unsloth\\\\studio\\\\auth\\\\llama_api_key_abc '
    '--mmproj C:\\\\models\\\\mmproj-F16.gguf"}')


def test_parse_unsloth_command_extracts_port_key_alias():
    parsed = parse_unsloth_llama_command(_UNSLOTH_LINE)
    assert parsed['port'] == 62541
    assert parsed['alias'] == 'DavidAU/Qwen3.8-27B-GGUF'
    assert parsed['api_key_file'].endswith('llama_api_key_abc')


def test_parse_unsloth_command_non_llama_line_is_none():
    assert parse_unsloth_llama_command('some other log line') is None
    assert parse_unsloth_llama_command('llama-server without port') is None


def test_detect_unsloth_endpoint_reads_latest_log_and_key(tmp_path):
    logs = tmp_path / 'logs'
    logs.mkdir()
    auth = tmp_path / 'auth'
    auth.mkdir()
    key_file = auth / 'llama_api_key_abc'
    key_file.write_text('secret-key-123\n')
    line = _UNSLOTH_LINE.replace(
        'C:\\\\Users\\\\me\\\\.unsloth\\\\studio\\\\auth\\\\llama_api_key_abc',
        str(key_file).replace('\\', '\\\\'))
    (logs / 'backend-old.log').write_text('Starting llama-server: --port 1111\n')
    (logs / 'backend-new.log').write_text(line + '\n')
    # Make the new log clearly newer.
    import os
    os.utime(logs / 'backend-old.log', (1, 1))
    endpoint = detect_unsloth_endpoint(studio_home=tmp_path)
    assert endpoint['base_url'] == 'http://127.0.0.1:62541/v1'
    assert endpoint['api_key'] == 'secret-key-123'
    assert endpoint['model'] == 'DavidAU/Qwen3.8-27B-GGUF'


def test_detect_unsloth_endpoint_missing_home_is_none(tmp_path):
    assert detect_unsloth_endpoint(studio_home=tmp_path / 'nope') is None


def test_detect_unsloth_endpoint_no_llama_log_is_none(tmp_path):
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'backend-x.log').write_text('nothing here\n')
    assert detect_unsloth_endpoint(studio_home=tmp_path) is None


# ---------------------------------------------------------------------------
# Dialog end-to-end: sweep -> judge -> apply winner
# ---------------------------------------------------------------------------

pytest.importorskip('PySide6', reason='PySide6 is not installed')

from utils.image import Image  # noqa: E402


class FakeCompareModel:
    is_remote_endpoint = False
    supports_batching = False
    batch_size = 1

    def __init__(self, captioning_thread_=None, caption_settings=None):
        self.caption_settings = caption_settings or {}
        self.prompt = self.caption_settings.get('prompt', '')

    def get_error_message(self):
        return None

    def load_processor_and_model(self):
        pass

    def monkey_patch_after_loading(self):
        pass

    def get_captioning_message(self, *args):
        return 'Fake comparing...'

    def get_image_prompt(self, image):
        return self.prompt

    def get_model_inputs(self, image_prompt, image):
        return {}

    def generate_caption(self, model_inputs, image_prompt):
        # Echo the prompt so different candidates produce different captions.
        return f'caption::{image_prompt}', f'caption::{image_prompt}'


class StubImageList:
    def __init__(self, model):
        self.model = model

    def get_selected_image_indices(self):
        return [self.model.index(0)]

    def select_image_indices(self, indices):
        pass


@pytest.fixture
def captioner(qtbot, isolated_settings):
    from models.image_list_model import ImageListModel
    from widgets.auto_captioner import AutoCaptioner
    model = ImageListModel(image_list_image_width=200, tag_separator=', ')
    path = Path(tempfile.gettempdir()) / 'taggui-ai-compare.png'
    if not path.exists():
        import PIL.Image as PilImage
        PilImage.new('RGB', (8, 8)).save(path)
    model.images = [Image(path=path, dimensions=(8, 8), tags=[])]
    captioner = AutoCaptioner(model, StubImageList(model))
    qtbot.addWidget(captioner)
    return captioner


def test_ai_comparison_end_to_end(captioner, qtbot, monkeypatch):
    import auto_captioning.captioning_thread as ct_module
    import dialogs.ai_comparison_dialog as dialog_module
    from dialogs.ai_comparison_dialog import AIComparisonDialog

    monkeypatch.setattr(ct_module, 'get_model_class',
                        lambda _model_id: FakeCompareModel)
    # Fake judge: rank candidate B best, with a parseable verdict.
    fake_verdict = ('{"overall": {"scores": {"A": 5, "B": 9}, '
                    '"ranking": ["B", "A"], "best": "B", '
                    '"reasoning": "B is better."}, "per_image": []}')
    monkeypatch.setattr(dialog_module, 'judge_captions',
                        lambda *args, **kwargs: fake_verdict)

    dialog = AIComparisonDialog(captioner)
    qtbot.addWidget(dialog)
    # Force the server judge path so the fake judge_captions is used.
    dialog.judge_source_combo.setCurrentText('OpenAI server (URL)')
    # Two prompt candidates (current + one purpose), current settings only.
    dialog.include_current_prompt_check.setChecked(True)
    captioner.caption_settings_form.prompt_text_edit.setPlainText('current prompt')
    for purpose, check in dialog.purpose_checks.items():
        check.setChecked(purpose == 'General')
    dialog.image_count_spin.setValue(1)
    assert len(dialog._prompt_candidates()) == 2

    dialog.start_or_stop()
    qtbot.waitUntil(lambda: dialog.verdict is not None, timeout=15000)
    assert dialog.verdict['best'] == 'B'
    assert dialog.apply_winner_button.isEnabled()
    # The results table has one row per sample image and a column per candidate.
    assert dialog.results_table.rowCount() == 1
    assert dialog.results_table.columnCount() == 1 + len(dialog.candidates)

    # Apply the winner loads its prompt into the form.
    winner = next(c for c in dialog.candidates if c['label'] == 'B')
    dialog.apply_winner()
    assert captioner.caption_settings_form.prompt_text_edit.toPlainText() == \
        winner['prompt']


def test_ai_comparison_judge_failure_still_shows_table(captioner, qtbot,
                                                       monkeypatch):
    import auto_captioning.captioning_thread as ct_module
    import dialogs.ai_comparison_dialog as dialog_module
    from dialogs.ai_comparison_dialog import AIComparisonDialog

    monkeypatch.setattr(ct_module, 'get_model_class',
                        lambda _model_id: FakeCompareModel)

    def failing_judge(*args, **kwargs):
        raise RuntimeError('connection refused')

    monkeypatch.setattr(dialog_module, 'judge_captions', failing_judge)
    dialog = AIComparisonDialog(captioner)
    qtbot.addWidget(dialog)
    dialog.judge_source_combo.setCurrentText('OpenAI server (URL)')
    dialog.include_current_prompt_check.setChecked(True)
    for purpose, check in dialog.purpose_checks.items():
        check.setChecked(purpose == 'General')
    dialog.image_count_spin.setValue(1)
    dialog.start_or_stop()
    qtbot.waitUntil(lambda: dialog.results_table.rowCount() > 0, timeout=15000)
    assert 'could not be reached' in dialog.verdict_text_edit.toPlainText()
    assert not dialog.apply_winner_button.isEnabled()
