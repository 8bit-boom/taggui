from pathlib import Path

from models.image_list_model import ImageListModel, Scope
from utils.image import Image

TAG_SEPARATOR = ', '


def make_model(tmp_path: Path, image_tags: list[list[str]]) -> ImageListModel:
    model = ImageListModel(image_list_image_width=200,
                           tag_separator=TAG_SEPARATOR)
    for i, tags in enumerate(image_tags):
        model.images.append(
            Image(path=tmp_path / f'image_{i}.jpg', dimensions=(100, 100),
                 tags=list(tags)))
    return model


def get_written_tags(image: Image) -> str:
    return image.path.with_suffix('.txt').read_text(encoding='utf-8')


def test_sort_tags_alphabetically(qapp, tmp_path):
    model = make_model(tmp_path, [['banana', 'apple', 'cherry']])
    model.sort_tags_alphabetically(do_not_reorder_first_tag=False)
    assert model.images[0].tags == ['apple', 'banana', 'cherry']
    assert get_written_tags(model.images[0]) == 'apple, banana, cherry'


def test_sort_tags_alphabetically_keeps_first_tag(qapp, tmp_path):
    model = make_model(tmp_path, [['zebra', 'banana', 'apple']])
    model.sort_tags_alphabetically(do_not_reorder_first_tag=True)
    assert model.images[0].tags == ['zebra', 'apple', 'banana']


def test_find_and_replace(qapp, tmp_path):
    model = make_model(tmp_path, [['a cat', 'a dog'], ['a bird']])
    model.find_and_replace('a ', '', scope=Scope.ALL_IMAGES, use_regex=False)
    assert model.images[0].tags == ['cat', 'dog']
    assert model.images[1].tags == ['bird']


def test_find_and_replace_with_regex(qapp, tmp_path):
    model = make_model(tmp_path, [['cat1', 'dog2']])
    model.find_and_replace(r'\d', '', scope=Scope.ALL_IMAGES, use_regex=True)
    assert model.images[0].tags == ['cat', 'dog']


def test_rename_tags(qapp, tmp_path):
    model = make_model(tmp_path, [['cat', 'dog'], ['bird']])
    model.rename_tags(['cat', 'bird'], 'animal')
    assert model.images[0].tags == ['animal', 'dog']
    assert model.images[1].tags == ['animal']


def test_delete_tags(qapp, tmp_path):
    model = make_model(tmp_path, [['cat', 'dog'], ['cat']])
    model.delete_tags(['cat'])
    assert model.images[0].tags == ['dog']
    assert model.images[1].tags == []


def test_remove_duplicate_tags(qapp, tmp_path):
    model = make_model(tmp_path, [['cat', 'dog', 'cat'], ['bird']])
    removed_count = model.remove_duplicate_tags()
    assert removed_count == 1
    assert model.images[0].tags == ['cat', 'dog']
    assert model.images[1].tags == ['bird']


def test_remove_empty_tags(qapp, tmp_path):
    model = make_model(tmp_path, [['cat', '', '  ', 'dog']])
    removed_count = model.remove_empty_tags()
    assert removed_count == 2
    assert model.images[0].tags == ['cat', 'dog']


def test_add_tags(qapp, tmp_path):
    model = make_model(tmp_path, [['cat'], ['dog']])
    image_indices = [model.index(0), model.index(1)]
    model.add_tags(['new'], image_indices)
    assert model.images[0].tags == ['cat', 'new']
    assert model.images[1].tags == ['dog', 'new']


def test_remove_tags(qapp, tmp_path):
    model = make_model(tmp_path, [['cat', 'new'], ['dog', 'new']])
    image_indices = [model.index(0), model.index(1)]
    model.remove_tags(['new'], image_indices)
    assert model.images[0].tags == ['cat']
    assert model.images[1].tags == ['dog']
    assert get_written_tags(model.images[0]) == 'cat'


def test_remove_tags_no_op_with_no_selected_images(qapp, tmp_path):
    model = make_model(tmp_path, [['cat']])
    model.remove_tags(['cat'], [])
    assert model.images[0].tags == ['cat']


def test_reverse_tags_order(qapp, tmp_path):
    model = make_model(tmp_path, [['a', 'b', 'c']])
    model.reverse_tags_order(do_not_reorder_first_tag=False)
    assert model.images[0].tags == ['c', 'b', 'a']


def test_move_tags_to_front(qapp, tmp_path):
    model = make_model(tmp_path, [['a', 'b', 'c']])
    model.move_tags_to_front(['c'])
    assert model.images[0].tags == ['c', 'a', 'b']


def test_update_image_tags(qapp, tmp_path):
    model = make_model(tmp_path, [['a']])
    model.update_image_tags(model.index(0), ['a', 'b'])
    assert model.images[0].tags == ['a', 'b']
    assert get_written_tags(model.images[0]) == 'a, b'
