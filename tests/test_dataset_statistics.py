from collections import Counter
from pathlib import Path

from utils.image import Image
from widgets.dataset_statistics import compute_statistics, get_aspect_ratio_bucket


class FakeBatchTokenizer:
    """A stand-in for the CLIP tokenizer's batched call signature."""

    def __call__(self, texts: list[str]):
        class Result:
            input_ids = [list(range(len(text.split()) + 2)) for text in texts]

        return Result()


def make_image(name: str, dimensions, tags: list[str],
               has_caption_file: bool = True, file_size: int = 1000) -> Image:
    return Image(path=Path(f'/dataset/{name}'), dimensions=dimensions,
                tags=tags, has_caption_file=has_caption_file,
                file_size=file_size)


def test_compute_statistics_empty_dataset():
    stats = compute_statistics([], Counter(), FakeBatchTokenizer(), ', ',
                               min_side=512)
    assert stats.image_count == 0
    assert stats.tags_per_image == (0.0, 0.0, 0.0, 0.0)


def test_compute_statistics_counts():
    images = [
        make_image('a.jpg', (1000, 1000), ['cat', 'dog']),
        make_image('b.jpg', (1000, 1000), []),
        make_image('c.jpg', (1000, 1000), ['cat'], has_caption_file=False),
    ]
    tag_counter = Counter()
    for image in images:
        tag_counter.update(image.tags)
    stats = compute_statistics(images, tag_counter, FakeBatchTokenizer(),
                               ', ', min_side=512)
    assert stats.image_count == 3
    assert stats.tagged_image_count == 2
    assert stats.untagged_image_count == 1
    assert stats.missing_caption_file_count == 1
    assert stats.unique_tag_count == 2
    assert stats.tag_instance_count == 3
    assert stats.singleton_tag_count == 1  # 'dog' appears once


def test_compute_statistics_duplicate_captions():
    images = [
        make_image('a.jpg', (100, 100), ['cat', 'dog']),
        make_image('b.jpg', (100, 100), ['cat', 'dog']),
        make_image('c.jpg', (100, 100), ['bird']),
    ]
    tag_counter = Counter()
    for image in images:
        tag_counter.update(image.tags)
    stats = compute_statistics(images, tag_counter, FakeBatchTokenizer(),
                               ', ', min_side=512)
    assert stats.duplicate_caption_groups == 1


def test_compute_statistics_small_images_and_megapixels():
    images = [
        make_image('a.jpg', (256, 256), ['cat']),
        make_image('b.jpg', (2000, 1000), ['dog']),
    ]
    tag_counter = Counter()
    for image in images:
        tag_counter.update(image.tags)
    stats = compute_statistics(images, tag_counter, FakeBatchTokenizer(),
                               ', ', min_side=512)
    assert stats.small_image_count == 1
    mean_mp, median_mp, min_mp, max_mp = stats.megapixels
    assert min_mp == 256 * 256 / 1_000_000
    assert max_mp == 2000 * 1000 / 1_000_000


def test_compute_statistics_total_bytes_ignores_missing_size():
    images = [
        make_image('a.jpg', (100, 100), [], file_size=1000),
        make_image('b.jpg', (100, 100), [], file_size=None),
    ]
    stats = compute_statistics(images, Counter(), FakeBatchTokenizer(), ', ',
                               min_side=512)
    assert stats.total_bytes == 1000


def test_get_aspect_ratio_bucket_square():
    assert get_aspect_ratio_bucket(1000, 1000) == '1:1'


def test_get_aspect_ratio_bucket_16_9():
    assert get_aspect_ratio_bucket(1920, 1080) == '16:9'


def test_get_aspect_ratio_bucket_other():
    assert get_aspect_ratio_bucket(1000, 333) == 'other'


def test_get_aspect_ratio_bucket_zero_height():
    assert get_aspect_ratio_bucket(1000, 0) == 'other'
