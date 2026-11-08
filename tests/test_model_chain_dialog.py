import tempfile
from pathlib import Path

import pytest

pytest.importorskip('PySide6', reason='PySide6 is not installed')

from PySide6.QtWidgets import QApplication

from utils.enums import CaptionPosition
from utils.image import Image


class FakeCaptionModel:
    """
    An instant, dependency-free stand-in for the selected captioning model,
    so the chain test exercises the real step -> thread -> next-step path
    without torch, transformers, or a GPU.
    """
    is_remote_endpoint = False
    supports_batching = True
    batch_size = 1

    def __init__(self, captioning_thread_=None, caption_settings=None):
        self.thread = captioning_thread_
        self.caption_settings = caption_settings

    def get_error_message(self):
        return None

    def load_processor_and_model(self):
        pass

    def monkey_patch_after_loading(self):
        pass

    def get_captioning_message(self, are_multiple_images_selected,
                               captioning_start_datetime):
        return 'Fake captioning...'

    def get_image_prompt(self, image):
        return 'Describe.'

    def load_image(self, image):
        import PIL.Image as PilImage
        return PilImage.new('RGB', (32, 32))

    def get_model_inputs(self, image_prompt, image):
        return {}

    def get_batched_model_inputs(self, batch):
        return batch

    def generate_captions_batch(self, model_inputs, batch):
        return ([f'step caption for {item["image_prompt"]}'
                for item in batch], [None] * len(batch))


class StubImageList:
    """Just the `AutoCaptioner`/`ModelChainDialog` entry points it needs."""
    def __init__(self, image_list_model):
        self.image_list_model = image_list_model

    def get_selected_image_indices(self):
        return [self.image_list_model.index(0)]

    def select_image_indices(self, indices):
        pass


@pytest.fixture
def fake_model_class(monkeypatch):
    import auto_captioning.captioning_thread as captioning_thread_module
    monkeypatch.setattr(captioning_thread_module, 'get_model_class',
                        lambda _model_id: FakeCaptionModel)


def build_captioner(qtbot, tmp_path):
    from models.image_list_model import ImageListModel
    from widgets.auto_captioner import AutoCaptioner

    image_list_model = ImageListModel(image_list_image_width=200,
                                      tag_separator=', ')
    image_path = tmp_path / 'taggui-chain-test.png'
    import PIL.Image as PilImage
    PilImage.new('RGB', (32, 32)).save(image_path)
    image_list_model.images = [Image(path=image_path, dimensions=(32, 32),
                                     tags=[])]
    captioner = AutoCaptioner(image_list_model,
                              StubImageList(image_list_model))
    qtbot.addWidget(captioner)
    return captioner, image_list_model


def test_chain_runs_steps_sequentially_and_applies_tags(
        qtbot, isolated_settings, fake_model_class, tmp_path):
    app = QApplication.instance() or QApplication([])
    from dialogs.model_chain_dialog import ModelChainDialog

    captioner, image_list_model = build_captioner(qtbot, tmp_path)
    # `MainWindow.connect_auto_captioner_signals()` is what actually applies
    # a generated caption to the image list in the real app; replicate just
    # that one connection, since this test builds `AutoCaptioner` directly
    # without a `MainWindow`.
    captioner.caption_generated.connect(
        lambda image_index, _caption, tags:
        image_list_model.update_image_tags(image_index, tags))
    dialog = ModelChainDialog(captioner)
    qtbot.addWidget(dialog)

    base_settings = captioner.caption_settings_form.get_caption_settings()
    step_1 = dict(base_settings)
    step_1['caption_position'] = CaptionPosition.OVERWRITE_ALL_TAGS
    step_1['model_id'] = 'fake-model-step-1'
    step_2 = dict(base_settings)
    step_2['caption_position'] = CaptionPosition.AFTER_LAST_TAG
    step_2['model_id'] = 'fake-model-step-2'
    dialog.steps = [step_1, step_2]
    dialog.refresh_steps_list()
    assert dialog.steps_list_widget.count() == 2

    dialog.start_or_stop()
    assert dialog.captioning_thread is not None
    assert dialog.run_stop_button.text() == 'Cancel Chain'

    # Both steps run and finish - with a fake, instant model, they can both
    # complete between two polls of `waitUntil`, so wait for the final
    # state directly rather than for `step_index` to pass through 1 (which
    # could be skipped over entirely between polls).
    qtbot.waitUntil(lambda: dialog.captioning_thread is None, timeout=10000)
    assert dialog.step_index == 2, (
        'The chain did not run both steps (step_index should end at 2, '
        'one past the last step).')
    assert dialog.run_stop_button.text() == 'Run Chain'
    assert dialog.close_button.isEnabled()
    assert captioner.start_cancel_button.isEnabled()

    image = image_list_model.images[0]
    # Step 1 overwrites all tags with its caption, step 2 appends its own
    # after - both steps' output should end up in the final tags.
    joined_tags = ', '.join(image.tags)
    assert 'step caption for Describe.' in joined_tags


def test_add_remove_move_steps(qtbot, isolated_settings, fake_model_class,
                               tmp_path):
    app = QApplication.instance() or QApplication([])
    from dialogs.model_chain_dialog import ModelChainDialog

    captioner, _ = build_captioner(qtbot, tmp_path)
    dialog = ModelChainDialog(captioner)
    qtbot.addWidget(dialog)

    captioner.caption_settings_form.model_combo_box.setCurrentText(
        captioner.caption_settings_form.model_combo_box.itemText(0))
    dialog.add_step()
    captioner.caption_settings_form.model_combo_box.setCurrentText(
        captioner.caption_settings_form.model_combo_box.itemText(1))
    dialog.add_step()
    assert len(dialog.steps) == 2
    first_model_id = dialog.steps[0]['model_id']
    second_model_id = dialog.steps[1]['model_id']
    assert first_model_id != second_model_id

    dialog.steps_list_widget.setCurrentRow(0)
    dialog.move_step(1)
    assert dialog.steps[0]['model_id'] == second_model_id
    assert dialog.steps[1]['model_id'] == first_model_id

    dialog.steps_list_widget.setCurrentRow(0)
    dialog.remove_step()
    assert len(dialog.steps) == 1
    assert dialog.steps[0]['model_id'] == first_model_id


def test_save_load_delete_chain(qtbot, isolated_settings, fake_model_class,
                                tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    from dialogs.model_chain_dialog import ModelChainDialog
    from auto_captioning.model_chains import get_model_chains

    captioner, _ = build_captioner(qtbot, tmp_path)
    dialog = ModelChainDialog(captioner)
    qtbot.addWidget(dialog)
    dialog.add_step()

    monkeypatch.setattr(
        'dialogs.model_chain_dialog.QInputDialog.getText',
        staticmethod(lambda *args, **kwargs: ('My Chain', True)))
    dialog.save_chain()
    assert 'My Chain' in get_model_chains()
    assert len(get_model_chains()['My Chain']) == 1

    dialog.steps = []
    dialog.refresh_steps_list()
    dialog.load_chain('My Chain')
    assert len(dialog.steps) == 1

    monkeypatch.setattr(
        'dialogs.model_chain_dialog.QMessageBox.question',
        staticmethod(lambda *args, **kwargs:
                     __import__('PySide6.QtWidgets', fromlist=['QMessageBox'])
                     .QMessageBox.StandardButton.Yes))
    dialog.chain_combo_box.setCurrentText('My Chain')
    dialog.delete_chain()
    assert 'My Chain' not in get_model_chains()
