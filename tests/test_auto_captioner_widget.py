import tempfile
from pathlib import Path

import pytest

pytest.importorskip('PySide6', reason='PySide6 is not installed')

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from utils.image import Image


class FakeCaptionModel:
    """
    An instant, dependency-free stand-in for the selected captioning model,
    so the widget test exercises the real button -> thread -> console path
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
        return ([f'caption {item["image_prompt"]}' for item in batch],
                [None] * len(batch))


class StubImageList:
    """Just the two `AutoCaptioner` entry points, without the widget."""
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


def test_start_button_starts_captioning(qtbot, isolated_settings,
                                        fake_model_class):
    """
    Regression test: the `Start Auto-Captioning` button's `clicked`
    connection was accidentally moved out of `__init__` during a refactor,
    so clicking it did nothing. This constructs the real dock and clicks
    the real button.
    """
    app = QApplication.instance() or QApplication([])
    from models.image_list_model import ImageListModel
    from widgets.auto_captioner import AutoCaptioner

    image_list_model = ImageListModel(image_list_image_width=200,
                                      tag_separator=', ')
    image_path = Path(tempfile.gettempdir()) / 'taggui-widget-test.png'
    if not image_path.exists():
        import PIL.Image as PilImage
        PilImage.new('RGB', (32, 32)).save(image_path)
    image_list_model.images = [Image(path=image_path, dimensions=(32, 32),
                                     tags=[])]
    captioner = AutoCaptioner(image_list_model, StubImageList(image_list_model))
    qtbot.addWidget(captioner)

    assert not captioner.is_captioning
    assert not captioner.caption_settings_form.unload_model_button.isEnabled()

    qtbot.mouseClick(captioner.start_cancel_button, __import__(
        'PySide6.QtCore', fromlist=['Qt']).Qt.MouseButton.LeftButton)
    # The click must flip the dock into the captioning state synchronously
    # (`generate_captions` sets it before starting the thread).
    assert captioner.is_captioning, (
        'Clicking Start did not start captioning; the clicked connection '
        'is missing or the handler returned early.')
    # Wait for the fake model to produce its caption in the console.
    qtbot.waitUntil(lambda: 'Fake captioning...' in
                    captioner.console_text_edit.toPlainText(), timeout=10000)
    # Then cancel/finish cleanly and wait for the thread to end.
    qtbot.waitUntil(lambda: not captioner.captioning_thread.isRunning(),
                    timeout=10000)
    assert not captioner.is_captioning
