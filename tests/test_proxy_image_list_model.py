from pathlib import Path

from models.image_list_model import ImageListModel
from models.proxy_image_list_model import ProxyImageListModel
from utils.image import Image

TAG_SEPARATOR = ', '


class FakeTokenizer:
    """A stand-in for the CLIP tokenizer: token count == word count."""

    def __call__(self, text: str):
        class Result:
            input_ids = list(range(len(text.split()) + 2))

        return Result()


def make_proxy_model() -> ProxyImageListModel:
    image_list_model = ImageListModel(image_list_image_width=200,
                                      tag_separator=TAG_SEPARATOR)
    return ProxyImageListModel(image_list_model, FakeTokenizer(),
                               TAG_SEPARATOR)


def make_image(path='/dataset/pic.jpg', dimensions=(1000, 500), tags=None,
              file_size=None) -> Image:
    return Image(path=Path(path), dimensions=dimensions, tags=tags or [],
                file_size=file_size)


def test_tag_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(tags=['cat', 'outdoors'])
    assert proxy_model.does_image_match_filter(image, ['tag', 'cat'])
    assert not proxy_model.does_image_match_filter(image, ['tag', 'dog'])


def test_tag_filter_wildcard(qapp):
    proxy_model = make_proxy_model()
    image = make_image(tags=['blue eyes'])
    assert proxy_model.does_image_match_filter(image, ['tag', 'blue*'])


def test_caption_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(tags=['cat', 'outdoors'])
    assert proxy_model.does_image_match_filter(image, ['caption', 'cat'])


def test_name_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(path='/dataset/subdir/my_pic.jpg')
    assert proxy_model.does_image_match_filter(image, ['name', 'my_pic'])


def test_path_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(path='/dataset/subdir/my_pic.jpg')
    assert proxy_model.does_image_match_filter(image, ['path', 'subdir'])


def test_ext_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(path='/dataset/pic.PNG')
    assert proxy_model.does_image_match_filter(image, ['ext', 'png'])
    assert not proxy_model.does_image_match_filter(image, ['ext', 'jpg'])


def test_not_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(tags=['cat'])
    assert proxy_model.does_image_match_filter(image, ['NOT', ['tag', 'dog']])
    assert not proxy_model.does_image_match_filter(
        image, ['NOT', ['tag', 'cat']])


def test_and_or_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(tags=['cat', 'outdoors'])
    assert proxy_model.does_image_match_filter(
        image, [['tag', 'cat'], 'AND', ['tag', 'outdoors']])
    assert not proxy_model.does_image_match_filter(
        image, [['tag', 'cat'], 'AND', ['tag', 'indoors']])
    assert proxy_model.does_image_match_filter(
        image, [['tag', 'cat'], 'OR', ['tag', 'indoors']])


def test_tags_count_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(tags=['a', 'b', 'c'])
    assert proxy_model.does_image_match_filter(image, ['tags', '=', '3'])
    assert proxy_model.does_image_match_filter(image, ['tags', '>', '2'])
    assert not proxy_model.does_image_match_filter(image, ['tags', '>', '3'])


def test_chars_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(tags=['abcde'])
    assert proxy_model.does_image_match_filter(image, ['chars', '=', '5'])


def test_tokens_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(tags=['one two three'])
    assert proxy_model.does_image_match_filter(image, ['tokens', '=', '3'])


def test_width_height_filters(qapp):
    proxy_model = make_proxy_model()
    image = make_image(dimensions=(1920, 1080))
    assert proxy_model.does_image_match_filter(image, ['width', '=', '1920'])
    assert proxy_model.does_image_match_filter(image, ['height', '<', '1100'])


def test_ratio_filter(qapp):
    proxy_model = make_proxy_model()
    image = make_image(dimensions=(1600, 800))
    assert proxy_model.does_image_match_filter(image, ['ratio', '=', '2'])


def test_area_and_mp_filters(qapp):
    proxy_model = make_proxy_model()
    image = make_image(dimensions=(1000, 1000))
    assert proxy_model.does_image_match_filter(image, ['area', '=', '1000000'])
    assert proxy_model.does_image_match_filter(image, ['mp', '=', '1'])


def test_size_filter_with_suffix(qapp):
    proxy_model = make_proxy_model()
    image = make_image(file_size=600_000)
    assert proxy_model.does_image_match_filter(image, ['size', '>', '500kb'])
    assert not proxy_model.does_image_match_filter(image,
                                                    ['size', '>', '1mb'])


def test_numeric_filter_with_missing_dimensions_does_not_match(qapp):
    proxy_model = make_proxy_model()
    image = make_image(dimensions=None)
    assert not proxy_model.does_image_match_filter(image,
                                                    ['width', '>', '0'])


def test_numeric_filter_with_missing_file_size_does_not_match(qapp):
    proxy_model = make_proxy_model()
    image = make_image(file_size=None)
    assert not proxy_model.does_image_match_filter(image, ['size', '>', '0'])


def test_parse_filter_number_suffixes():
    assert ProxyImageListModel.parse_filter_number('500') == 500
    assert ProxyImageListModel.parse_filter_number('500kb') == 500_000
    assert ProxyImageListModel.parse_filter_number('1.5mb') == 1_500_000
    assert ProxyImageListModel.parse_filter_number('2gb') == 2_000_000_000
