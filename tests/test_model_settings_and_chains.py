import tempfile
from pathlib import Path

import pytest

pytest.importorskip('PySide6', reason='PySide6 is not installed')

from auto_captioning.model_chains import get_model_chains, save_model_chains
from auto_captioning.model_settings import (get_model_settings,
                                             save_model_settings)
from utils.enums import CaptionPosition
from utils.image import Image


def test_model_settings_round_trip(isolated_settings):
    assert get_model_settings() == {}
    model_settings = {'Qwen/Qwen3-VL-8B-Instruct': {'prompt': 'Describe.'}}
    save_model_settings(model_settings)
    assert get_model_settings() == model_settings


def test_model_settings_default_on_corrupt_value(isolated_settings,
                                                  monkeypatch):
    from utils.settings import get_settings
    get_settings().setValue('model_settings', 'not valid json')
    assert get_model_settings() == {}
    get_settings().setValue('model_settings', '["a", "list", "not a dict"]')
    assert get_model_settings() == {}


def test_model_chains_round_trip(isolated_settings):
    assert get_model_chains() == {}
    chains = {
        'WD Tagger then Qwen': [
            {'model_id': 'SmilingWolf/wd-vit-tagger-v3',
             'caption_position': 'Overwrite'},
            {'model_id': 'Qwen/Qwen3-VL-8B-Instruct',
             'caption_position': 'Append to end',
             'prompt': 'Describe this image using the tags: {tags}'},
        ]
    }
    save_model_chains(chains)
    assert get_model_chains() == chains
    assert len(get_model_chains()['WD Tagger then Qwen']) == 2


def test_model_chains_default_on_corrupt_value(isolated_settings):
    from utils.settings import get_settings
    get_settings().setValue('model_chains', 'not valid json')
    assert get_model_chains() == {}


# ---------------------------------------------------------------------------
# describe_step (chain step summary shown in the dialog list)
# ---------------------------------------------------------------------------

def test_describe_step_includes_model_position_and_prompt():
    from dialogs.model_chain_dialog import describe_step
    step = {'model_id': 'Qwen/Qwen3-VL-8B-Instruct',
            'caption_position': 'Overwrite all tags',
            'prompt': 'Describe this image.'}
    text = describe_step(step, 1)
    assert text.startswith('1. Qwen/Qwen3-VL-8B-Instruct')
    assert 'Overwrite all tags' in text
    assert 'Describe this image.' in text


def test_describe_step_truncates_long_prompt_and_flattens_newlines():
    from dialogs.model_chain_dialog import describe_step
    text = describe_step(
        {'model_id': 'm', 'caption_position': '', 'prompt': 'word ' * 40}, 2)
    assert '...' in text
    assert '\n' not in text
    multiline = describe_step(
        {'model_id': 'm', 'prompt': 'line one\nline two'}, 1)
    assert '\n' not in multiline


def test_describe_step_handles_missing_fields():
    from dialogs.model_chain_dialog import describe_step
    assert '(unknown model)' in describe_step({}, 1)


# ---------------------------------------------------------------------------
# Widget fixtures
# ---------------------------------------------------------------------------

class StubImageList:
    def __init__(self, image_list_model):
        self.image_list_model = image_list_model

    def get_selected_image_indices(self):
        return [self.image_list_model.index(0)]

    def select_image_indices(self, indices):
        pass


@pytest.fixture
def captioner(qtbot, isolated_settings):
    from models.image_list_model import ImageListModel
    from widgets.auto_captioner import AutoCaptioner

    image_list_model = ImageListModel(image_list_image_width=200,
                                      tag_separator=', ')
    image_path = Path(tempfile.gettempdir()) / 'taggui-chain-test.png'
    if not image_path.exists():
        import PIL.Image as PilImage
        PilImage.new('RGB', (32, 32)).save(image_path)
    image_list_model.images = [Image(path=image_path, dimensions=(32, 32),
                                     tags=[])]
    captioner = AutoCaptioner(image_list_model, StubImageList(image_list_model))
    qtbot.addWidget(captioner)
    return captioner


# ---------------------------------------------------------------------------
# Per-model settings: switching models saves outgoing / restores incoming
# ---------------------------------------------------------------------------

def test_on_model_changed_saves_outgoing_under_old_id(captioner):
    # The combo has already changed to the new model by the time the slot
    # runs, so the outgoing snapshot must be stored under the OLD model id
    # (not the new one the combo now reports).
    form = captioner.caption_settings_form
    form.model_combo_box.setCurrentText('test-model-A')
    form.on_model_changed('test-model-A')
    form.prompt_text_edit.setPlainText('prompt for A')

    form.model_combo_box.setCurrentText('test-model-B')
    form.on_model_changed('test-model-B')

    saved = get_model_settings()
    assert 'test-model-A' in saved
    assert saved['test-model-A']['model_id'] == 'test-model-A'
    assert saved['test-model-A']['prompt'] == 'prompt for A'
    assert 'test-model-B' not in saved


def test_on_model_changed_restores_incoming_models_settings(captioner):
    form = captioner.caption_settings_form
    form.model_combo_box.setCurrentText('test-model-A')
    form.on_model_changed('test-model-A')
    form.prompt_text_edit.setPlainText('remembered A prompt')
    form.model_combo_box.setCurrentText('test-model-B')
    form.on_model_changed('test-model-B')
    form.prompt_text_edit.setPlainText('temporary B prompt')
    form.model_combo_box.setCurrentText('test-model-A')
    form.on_model_changed('test-model-A')
    assert form.prompt_text_edit.toPlainText() == 'remembered A prompt'
    assert get_model_settings()['test-model-B']['prompt'] == \
        'temporary B prompt'


def test_on_model_changed_same_model_is_noop(captioner):
    form = captioner.caption_settings_form
    form.model_combo_box.setCurrentText('test-model-A')
    form.on_model_changed('test-model-A')  # default -> A (saves the default)
    form.prompt_text_edit.setPlainText('keep me')
    before = get_model_settings()
    form.on_model_changed('test-model-A')  # A -> A: no save/restore
    assert form.prompt_text_edit.toPlainText() == 'keep me'
    assert get_model_settings() == before
    assert 'test-model-A' not in get_model_settings()


# ---------------------------------------------------------------------------
# Model Chain dialog: step editing
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_dialog(captioner, qtbot):
    from dialogs.model_chain_dialog import ModelChainDialog
    dialog = ModelChainDialog(captioner)
    qtbot.addWidget(dialog)
    return dialog


def test_chain_add_remove_move_steps(chain_dialog, captioner):
    form = captioner.caption_settings_form
    dialog = chain_dialog
    form.prompt_text_edit.setPlainText('step one')
    dialog.add_step()
    form.prompt_text_edit.setPlainText('step two')
    dialog.add_step()
    assert len(dialog.steps) == 2
    assert dialog.steps[0]['prompt'] == 'step one'
    assert dialog.steps[1]['prompt'] == 'step two'

    dialog.steps_list_widget.setCurrentRow(1)
    dialog.move_step(-1)
    assert dialog.steps[0]['prompt'] == 'step two'
    assert dialog.steps[1]['prompt'] == 'step one'

    dialog.steps_list_widget.setCurrentRow(0)
    dialog.remove_step()
    assert len(dialog.steps) == 1
    assert dialog.steps[0]['prompt'] == 'step one'


def test_chain_move_out_of_range_is_noop(chain_dialog, captioner):
    captioner.caption_settings_form.prompt_text_edit.setPlainText('only')
    chain_dialog.add_step()
    chain_dialog.steps_list_widget.setCurrentRow(0)
    chain_dialog.move_step(-1)
    chain_dialog.move_step(1)
    assert len(chain_dialog.steps) == 1


def test_chain_save_and_load_round_trip(chain_dialog, captioner, monkeypatch):
    form = captioner.caption_settings_form
    form.prompt_text_edit.setPlainText('chained prompt')
    chain_dialog.add_step()
    monkeypatch.setattr(
        'dialogs.model_chain_dialog.QInputDialog.getText',
        staticmethod(lambda *a, **k: ('my-chain', True)))
    chain_dialog.save_chain()
    assert 'my-chain' in get_model_chains()
    chain_dialog.steps = []
    chain_dialog.load_chain('my-chain')
    assert len(chain_dialog.steps) == 1
    assert chain_dialog.steps[0]['prompt'] == 'chained prompt'


# ---------------------------------------------------------------------------
# Model Chain run: a two-step pipeline where step 2 reads step 1's output
# ---------------------------------------------------------------------------

class FakeChainModel:
    """Per-image fake whose caption echoes its model id + substituted prompt.

    Uses the real {tags} template substitution, so a later step's prompt can
    reference an earlier step's applied caption - the core Model Chains use
    case (WD Tagger tags -> natural-language caption).
    """
    is_remote_endpoint = False
    supports_batching = False
    batch_size = 1

    def __init__(self, captioning_thread_=None, caption_settings=None):
        self.thread = captioning_thread_
        self.caption_settings = caption_settings or {}
        self.model_id = self.caption_settings.get('model_id', 'fake')
        self.prompt = self.caption_settings.get('prompt', '')

    def get_error_message(self):
        return None

    def load_processor_and_model(self):
        pass

    def monkey_patch_after_loading(self):
        pass

    def get_captioning_message(self, *args):
        return f'Fake {self.model_id} captioning...'

    def get_image_prompt(self, image):
        from auto_captioning.auto_captioning_model import (
            replace_template_variables)
        return replace_template_variables(self.prompt, image)

    def get_model_inputs(self, image_prompt, image):
        return {}

    def generate_caption(self, model_inputs, image_prompt):
        caption = f'{self.model_id}|{image_prompt}'
        return caption, caption


def test_chain_run_applies_steps_in_order_with_tags_reference(
        captioner, qtbot, monkeypatch):
    import auto_captioning.captioning_thread as ct_module
    from dialogs.model_chain_dialog import ModelChainDialog

    monkeypatch.setattr(ct_module, 'get_model_class',
                        lambda _model_id: FakeChainModel)
    image_list_model = captioner.image_list_model
    captioner.caption_generated.connect(
        lambda image_index, _caption, tags:
        image_list_model.update_image_tags(image_index, tags))

    base = captioner.caption_settings_form.get_caption_settings()
    step1 = {**base, 'model_id': 'tagger', 'prompt': 'booru',
             'caption_position': CaptionPosition.OVERWRITE_ALL_TAGS}
    step2 = {**base, 'model_id': 'captioner', 'prompt': 'Describe {tags}',
             'caption_position': CaptionPosition.OVERWRITE_ALL_TAGS}

    dialog = ModelChainDialog(captioner)
    qtbot.addWidget(dialog)
    dialog.steps = [step1, step2]
    dialog.selected_image_indices = [image_list_model.index(0)]
    dialog.step_index = 0
    dialog.start_step()

    qtbot.waitUntil(
        lambda: dialog.step_index >= len(dialog.steps)
        and dialog.captioning_thread is None, timeout=15000)

    tags = image_list_model.images[0].tags
    # step1 caption "tagger|booru" -> tags ["tagger|booru"]; step2 prompt
    # "Describe {tags}" -> "Describe tagger|booru" -> caption
    # "captioner|Describe tagger|booru". Proves order + {tags} referencing.
    assert tags == ['captioner|Describe tagger|booru'], tags


def test_chain_run_single_step_overwrites_tags(captioner, qtbot, monkeypatch):
    import auto_captioning.captioning_thread as ct_module
    from dialogs.model_chain_dialog import ModelChainDialog

    monkeypatch.setattr(ct_module, 'get_model_class',
                        lambda _model_id: FakeChainModel)
    image_list_model = captioner.image_list_model
    captioner.caption_generated.connect(
        lambda image_index, _caption, tags:
        image_list_model.update_image_tags(image_index, tags))
    base = captioner.caption_settings_form.get_caption_settings()
    step = {**base, 'model_id': 'solo-model', 'prompt': 'hello',
            'caption_position': CaptionPosition.OVERWRITE_ALL_TAGS}
    dialog = ModelChainDialog(captioner)
    qtbot.addWidget(dialog)
    dialog.steps = [step]
    dialog.selected_image_indices = [image_list_model.index(0)]
    dialog.step_index = 0
    dialog.start_step()
    qtbot.waitUntil(
        lambda: dialog.step_index >= 1 and dialog.captioning_thread is None,
        timeout=15000)
    assert image_list_model.images[0].tags == ['solo-model|hello']
