"""
Tests for in-app local GGUF inference (no server): the CUDA DLL path helper,
GGUF discovery, the cached loader/judge, and the AI-comparison dialog using a
local in-app judge.
"""

import tempfile
from pathlib import Path

import pytest

from auto_captioning import local_llama


# ---------------------------------------------------------------------------
# GGUF discovery
# ---------------------------------------------------------------------------

def test_discover_gguf_models_excludes_mmproj_and_dedups(tmp_path):
    (tmp_path / 'model-a.gguf').write_bytes(b'')
    (tmp_path / 'model-b.gguf').write_bytes(b'')
    (tmp_path / 'mmproj-F16.gguf').write_bytes(b'')  # projector, excluded
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'model-a.gguf').write_bytes(b'')  # same name, different dir
    found = local_llama.discover_gguf_models([tmp_path])
    names = [p.name for p in found]
    assert 'mmproj-F16.gguf' not in names
    assert names.count('model-a.gguf') == 2  # distinct paths kept
    assert 'model-b.gguf' in names
    assert found == sorted(found)


def test_discover_gguf_models_missing_dir_is_tolerated(tmp_path):
    # The HF cache is always scanned; a missing extra dir must not crash nor
    # change the result compared to passing no extra dirs.
    without = local_llama.discover_gguf_models([])
    with_missing = local_llama.discover_gguf_models([tmp_path / 'nope'])
    assert with_missing == without


# ---------------------------------------------------------------------------
# CUDA DLL path helper
# ---------------------------------------------------------------------------

def test_ensure_cuda_dll_path_noop_on_non_windows(monkeypatch):
    monkeypatch.setattr(local_llama.sys, 'platform', 'linux')
    # Should return without touching os.add_dll_directory.
    called = []
    # `os.add_dll_directory` only exists on Windows.
    monkeypatch.setattr(local_llama.os, 'add_dll_directory',
                        lambda *a: called.append(a), raising=False)
    local_llama.ensure_cuda_dll_path()
    assert called == []


def test_ensure_cuda_dll_path_tolerates_missing_torch(monkeypatch):
    monkeypatch.setattr(local_llama.sys, 'platform', 'win32')
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'torch':
            raise ImportError('no torch')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    local_llama.ensure_cuda_dll_path()  # must not raise


# ---------------------------------------------------------------------------
# Local judge (mocked model)
# ---------------------------------------------------------------------------

class FakeLlama:
    def __init__(self, content='{"overall": {"best": "A"}}'):
        self.content = content
        self.calls = []

    def create_chat_completion(self, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        return {'choices': [{'message': {'content': self.content}}]}


def test_judge_captions_local_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match='No local GGUF model'):
        local_llama.judge_captions_local(str(tmp_path / 'absent.gguf'), 'prompt')


def test_judge_captions_local_returns_model_content(tmp_path, monkeypatch):
    gguf = tmp_path / 'model.gguf'
    gguf.write_bytes(b'')
    fake = FakeLlama('verdict text')
    monkeypatch.setattr(local_llama, 'load_local_model',
                        lambda *a, **k: fake)
    result = local_llama.judge_captions_local(str(gguf), 'judge prompt')
    assert result == 'verdict text'
    # Greedy decoding and the prompt passed through.
    assert fake.calls[0][2] == 0
    assert fake.calls[0][0][0]['content'] == 'judge prompt'


def test_judge_captions_local_bad_response_raises(tmp_path, monkeypatch):
    gguf = tmp_path / 'model.gguf'
    gguf.write_bytes(b'')

    class BadModel:
        def create_chat_completion(self, **kwargs):
            return {'unexpected': True}

    monkeypatch.setattr(local_llama, 'load_local_model',
                        lambda *a, **k: BadModel())
    with pytest.raises(RuntimeError, match='unexpected response'):
        local_llama.judge_captions_local(str(gguf), 'prompt')


def test_load_local_model_missing_llama_cpp_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(local_llama, 'ensure_cuda_dll_path', lambda: None)
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'llama_cpp':
            raise ImportError('missing')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    local_llama.release_local_models()
    with pytest.raises(RuntimeError, match='llama-cpp-python is not installed'):
        local_llama.load_local_model(str(tmp_path / 'm.gguf'))


def test_release_local_models_clears_cache():
    local_llama._LOADED_MODELS[('x', -1, 8192)] = object()
    local_llama.release_local_models()
    assert local_llama._LOADED_MODELS == {}


# ---------------------------------------------------------------------------
# Dialog using the in-app local judge
# ---------------------------------------------------------------------------

pytest.importorskip('PySide6', reason='PySide6 is not installed')

from utils.image import Image  # noqa: E402


class FakeCompareModel:
    is_remote_endpoint = False
    supports_batching = False
    batch_size = 1

    def __init__(self, captioning_thread_=None, caption_settings=None):
        self.prompt = (caption_settings or {}).get('prompt', '')

    def get_error_message(self):
        return None

    def load_processor_and_model(self):
        pass

    def monkey_patch_after_loading(self):
        pass

    def get_captioning_message(self, *a):
        return 'fake'

    def get_image_prompt(self, image):
        return self.prompt

    def get_model_inputs(self, p, i):
        return {}

    def generate_caption(self, mi, p):
        return f'c::{p}', f'c::{p}'


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
    path = Path(tempfile.gettempdir()) / 'taggui-local-judge.png'
    if not path.exists():
        import PIL.Image as PilImage
        PilImage.new('RGB', (8, 8)).save(path)
    model.images = [Image(path=path, dimensions=(8, 8), tags=[])]
    captioner = AutoCaptioner(model, StubImageList(model))
    qtbot.addWidget(captioner)
    return captioner


def test_dialog_local_gguf_judge_end_to_end(captioner, qtbot, monkeypatch,
                                            tmp_path):
    import auto_captioning.captioning_thread as ct_module
    from dialogs.ai_comparison_dialog import AIComparisonDialog

    monkeypatch.setattr(ct_module, 'get_model_class',
                        lambda _model_id: FakeCompareModel)
    fake_verdict = ('{"overall": {"best": "B", "ranking": ["B", "A"], '
                    '"scores": {"A": 4, "B": 9}, "reasoning": "B wins"}}')
    monkeypatch.setattr(local_llama, 'judge_captions_local',
                        lambda *a, **k: fake_verdict)

    dialog = AIComparisonDialog(captioner)
    qtbot.addWidget(dialog)
    dialog.judge_source_combo.setCurrentText('Local GGUF (in-app, no server)')
    gguf = tmp_path / 'judge.gguf'
    gguf.write_bytes(b'')
    dialog.gguf_combo.setCurrentText(str(gguf))
    dialog.include_current_prompt_check.setChecked(True)
    captioner.caption_settings_form.prompt_text_edit.setPlainText(
        'current prompt')
    for purpose, check in dialog.purpose_checks.items():
        check.setChecked(purpose == 'General')
    dialog.image_count_spin.setValue(1)
    # Local widgets shown, server widgets hidden (dialog not exec'd, so use
    # isHidden rather than isVisible).
    assert not dialog.local_judge_widget.isHidden()
    assert dialog.server_judge_widget.isHidden()

    dialog.start_or_stop()
    qtbot.waitUntil(lambda: dialog.verdict is not None, timeout=15000)
    assert dialog.verdict['best'] == 'B'
    assert dialog.apply_winner_button.isEnabled()
