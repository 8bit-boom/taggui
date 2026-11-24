from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image as PILImage

from auto_captioning.captioning_thread import CaptioningThread
from utils.enums import CaptionPosition


class BatchModel:
    supports_batching = True
    is_remote_endpoint = False
    group_similar_images = True

    def __init__(self, batch_size=4, maximum=100):
        self.batch_size = batch_size
        self.maximum = maximum
        self.calls = []
        self.caption_settings = {'min_image_megapixels': 0.1,
                                 'max_image_megapixels': 1.0}
        self.device = torch.device('cpu')

    def get_pixels(self):
        return 100_000, 1_000_000

    def get_image_prompt(self, image):
        return image.path.stem

    def load_image(self, image):
        return PILImage.new('RGB', image.dimensions)

    def get_model_inputs(self, prompt, image):
        return prompt

    def generate_caption(self, inputs, prompt):
        return prompt, prompt

    def get_batched_model_inputs(self, batch):
        self.calls.append([item['image_prompt'] for item in batch])
        if len(batch) > self.maximum:
            raise torch.cuda.OutOfMemoryError('test')
        return batch

    def generate_captions_batch(self, inputs, batch):
        captions = [item['image_prompt'] for item in batch]
        return captions, captions


def make_thread(qapp, images):
    class ImageList:
        def data(self, index, role):
            return images[index]

    return CaptioningThread(None, ImageList(), list(range(len(images))),
                            {}, ', ', None)


def make_images(sizes):
    return [SimpleNamespace(path=Path(f'image{i}.png'), dimensions=size,
                            tags=[]) for i, size in enumerate(sizes)]


def test_oom_reduces_batch_and_retains_size(qapp, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    images = make_images([(10, 10)] * 6)
    thread = make_thread(qapp, images)
    model = BatchModel(batch_size=4, maximum=2)
    batch = [(i, image, model.get_image_prompt(image), model.load_image(image))
             for i, image in enumerate(images[:4])]
    results = thread.generate_batch_results(model, batch)
    assert [result[0] for result in results] == [
        'image0', 'image1', 'image2', 'image3']
    assert list(map(len, model.calls)) == [4, 2, 2]
    assert model.batch_size == 2
    thread.generate_batch_results(model, batch[:2])
    assert len(model.calls[-1]) == 2


def test_non_oom_falls_back_without_reducing(qapp):
    images = make_images([(10, 10)] * 2)
    thread = make_thread(qapp, images)
    model = BatchModel(2)
    def fail(_batch):
        raise RuntimeError('not an out of memory condition')
    model.get_batched_model_inputs = fail
    batch = [(i, image, model.get_image_prompt(image), model.load_image(image))
             for i, image in enumerate(images)]
    assert [r[0] for r in thread.generate_batch_results(model, batch)] == [
        'image0', 'image1']
    assert model.batch_size == 2


def test_oom_at_one_uses_single_image_retry(qapp, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    images = make_images([(10, 10)] * 2)
    thread = make_thread(qapp, images)
    model = BatchModel(2, maximum=0)
    batch = [(i, image, model.get_image_prompt(image), model.load_image(image))
             for i, image in enumerate(images)]
    assert [r[0] for r in thread.generate_batch_results(model, batch)] == [
        'image0', 'image1']
    assert model.batch_size == 1
    assert list(map(len, model.calls)) == [2, 1, 1]


def test_incomplete_batch_output_falls_back(qapp):
    images = make_images([(10, 10)] * 2)
    thread = make_thread(qapp, images)
    model = BatchModel(2)
    model.generate_captions_batch = lambda inputs, batch: (['partial'], ['partial'])
    batch = [(i, image, model.get_image_prompt(image), model.load_image(image))
             for i, image in enumerate(images)]
    assert [r[0] for r in thread.generate_batch_results(model, batch)] == [
        'image0', 'image1']
    assert model.batch_size == 2


def test_grouping_restores_original_association(qapp, monkeypatch):
    from PySide6.QtCore import QAbstractListModel, QModelIndex

    images = make_images([(100, 100), (10, 10), (90, 90), (12, 12)])
    class ImageList(QAbstractListModel):
        def rowCount(self, parent=QModelIndex()):
            return len(images)

        def data(self, index, role):
            return images[index.row()]

    image_list = ImageList()
    indices = [image_list.index(i) for i in range(len(images))]
    thread = CaptioningThread(None, image_list, indices, {}, ', ', None)
    thread.captioning_start_time = 0
    thread.total_target_count = len(indices)
    model = BatchModel(2)
    emitted = []
    thread.caption_generated.connect(
        lambda index, caption, tags: emitted.append((index.row(), caption)))
    thread._run_captioning_loop(model, indices, CaptionPosition.DO_NOT_ADD,
                                False)
    assert model.calls == [['image1', 'image3'], ['image2', 'image0']]
    assert emitted == [(i, f'image{i}') for i in range(4)]


def test_pilot_stops_before_grouped_window_overshoots(qapp):
    from PySide6.QtCore import QAbstractListModel, QModelIndex

    images = make_images([(100, 100), (10, 10), (90, 90), (12, 12)])
    class ImageList(QAbstractListModel):
        def rowCount(self, parent=QModelIndex()):
            return len(images)

        def data(self, index, role):
            return images[index.row()]

    image_list = ImageList()
    indices = [image_list.index(i) for i in range(4)]
    thread = CaptioningThread(None, image_list, indices, {}, ', ', None)
    thread.pilot_run_count = 2
    thread.continue_after_pilot = False
    emitted = []
    prompted = []
    thread.caption_generated.connect(
        lambda index, caption, tags: emitted.append(index.row()))
    thread.pilot_run_finished_requested.connect(prompted.append)
    thread._run_captioning_loop(BatchModel(2), indices,
                                CaptionPosition.DO_NOT_ADD, False)
    assert emitted == [0, 1]
    assert prompted == [2]


def test_cancel_keeps_completed_groups(qapp):
    from PySide6.QtCore import QAbstractListModel, QModelIndex

    images = make_images([(100, 100), (10, 10), (90, 90), (12, 12)])
    class ImageList(QAbstractListModel):
        def rowCount(self, parent=QModelIndex()):
            return len(images)

        def data(self, index, role):
            return images[index.row()]

    image_list = ImageList()
    indices = [image_list.index(i) for i in range(4)]
    thread = CaptioningThread(None, image_list, indices, {}, ', ', None)
    model = BatchModel(2)
    generate = model.generate_captions_batch
    def cancel_second(inputs, batch):
        if len(model.calls) == 2:
            thread.is_canceled = True
        return generate(inputs, batch)
    model.generate_captions_batch = cancel_second
    emitted = []
    thread.caption_generated.connect(
        lambda index, caption, tags: emitted.append((index.row(), caption)))
    thread._run_captioning_loop(model, indices, CaptionPosition.DO_NOT_ADD,
                                False)
    assert emitted == [(1, 'image1'), (3, 'image3')]


def test_benchmark_discards_results_and_restores_settings(qapp, capsys):
    images = make_images([(10, 10)] * 4)
    thread = make_thread(qapp, images)
    model = BatchModel(4)
    original = model.caption_settings
    thread.benchmark(model, list(range(4)))
    output = capsys.readouterr().out
    assert 'batch/concurrency 1' in output
    assert 'batch/concurrency 2' in output
    assert 'batch/concurrency 4' in output
    assert '0.50 MP' in output
    assert 'recommendation' in output
    assert model.caption_settings is original
    assert model.batch_size == 4
    assert all(not image.tags for image in images)
    assert thread.captioned_total == 0


def test_benchmark_uses_grouped_windows(qapp):
    images = make_images([(100, 100), (10, 10), (90, 90), (12, 12)])
    thread = make_thread(qapp, images)
    model = BatchModel(2)
    thread.benchmark(model, list(range(4)))
    assert ['image1', 'image3'] in model.calls
    assert ['image2', 'image0'] in model.calls


def test_remote_benchmark_limits_concurrency_to_user_setting(qapp, capsys):
    images = make_images([(10, 10)] * 8)
    thread = make_thread(qapp, images)
    class RemoteModel(BatchModel):
        supports_batching = False
        is_remote_endpoint = True
        group_similar_images = False

        def get_model_inputs(self, prompt, image):
            return prompt

        def generate_caption(self, inputs, prompt):
            return prompt, prompt

    model = RemoteModel(3)
    thread.benchmark(model, list(range(8)))
    output = capsys.readouterr().out
    assert 'batch/concurrency 1' in output
    assert 'batch/concurrency 3' in output
    assert 'batch/concurrency 4' not in output
    assert 'remote server not measured' in output
    assert model.batch_size == 3


def test_benchmark_cancellation_restores_settings(qapp):
    images = make_images([(10, 10)] * 4)
    thread = make_thread(qapp, images)
    model = BatchModel(4)
    original = model.caption_settings
    def cancel(batch):
        thread.is_canceled = True
        return batch
    model.get_batched_model_inputs = cancel
    thread.benchmark(model, list(range(4)))
    assert model.caption_settings is original
    assert model.batch_size == 4
