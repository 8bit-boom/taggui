"""
Tests for precise tag insertion: the shared engine, auto-caption "Insert at
position", and the model-level multi-image add with insertion modes.
"""

import pytest

from utils.tag_insertion import TagInsertMode, insert_tags


# ---------------------------------------------------------------------------
# insert_tags engine
# ---------------------------------------------------------------------------

def test_top_and_bottom():
    existing = ['a', 'b', 'c']
    assert insert_tags(existing, ['x'], TagInsertMode.TOP) == \
        ['x', 'a', 'b', 'c']
    assert insert_tags(existing, ['x'], TagInsertMode.BOTTOM) == \
        ['a', 'b', 'c', 'x']


@pytest.mark.parametrize('index,expected', [
    (0, ['x', 'a', 'b', 'c']),   # top
    (1, ['a', 'x', 'b', 'c']),   # between a and b
    (2, ['a', 'b', 'x', 'c']),   # between b and c
    (3, ['a', 'b', 'c', 'x']),   # bottom
])
def test_at_index_inserts_between_tags(index, expected):
    assert insert_tags(['a', 'b', 'c'], ['x'], TagInsertMode.AT_INDEX,
                       index) == expected


def test_at_index_clamps_out_of_range():
    existing = ['a', 'b']
    assert insert_tags(existing, ['x'], TagInsertMode.AT_INDEX, -5) == \
        ['x', 'a', 'b']
    assert insert_tags(existing, ['x'], TagInsertMode.AT_INDEX, 99) == \
        ['a', 'b', 'x']


def test_at_index_multiple_new_tags_keep_order():
    assert insert_tags(['a', 'c'], ['x', 'y'], TagInsertMode.AT_INDEX, 1) == \
        ['a', 'x', 'y', 'c']


def test_before_and_after_selection():
    existing = ['a', 'b', 'c', 'd']
    assert insert_tags(existing, ['x'], TagInsertMode.BEFORE_SELECTED,
                       selected_rows=[2]) == ['a', 'b', 'x', 'c', 'd']
    assert insert_tags(existing, ['x'], TagInsertMode.AFTER_SELECTED,
                       selected_rows=[2]) == ['a', 'b', 'c', 'x', 'd']


def test_selection_relative_uses_span_of_multi_selection():
    existing = ['a', 'b', 'c', 'd', 'e']
    # Before the first selected row (1), after the last selected row (3).
    assert insert_tags(existing, ['x'], TagInsertMode.BEFORE_SELECTED,
                       selected_rows=[1, 3]) == ['a', 'x', 'b', 'c', 'd', 'e']
    assert insert_tags(existing, ['x'], TagInsertMode.AFTER_SELECTED,
                       selected_rows=[1, 3]) == ['a', 'b', 'c', 'd', 'x', 'e']


def test_replace_selection():
    existing = ['a', 'b', 'c', 'd']
    assert insert_tags(existing, ['x'], TagInsertMode.REPLACE_SELECTED,
                       selected_rows=[1, 2]) == ['a', 'x', 'd']
    assert insert_tags(existing, ['x', 'y'], TagInsertMode.REPLACE_SELECTED,
                       selected_rows=[0]) == ['x', 'y', 'b', 'c', 'd']


def test_selection_modes_fall_back_to_bottom_without_selection():
    existing = ['a', 'b']
    for mode in (TagInsertMode.BEFORE_SELECTED, TagInsertMode.AFTER_SELECTED,
                 TagInsertMode.REPLACE_SELECTED):
        assert insert_tags(existing, ['x'], mode, selected_rows=[]) == \
            ['a', 'b', 'x']
        assert insert_tags(existing, ['x'], mode) == ['a', 'b', 'x']


def test_out_of_range_selection_rows_ignored():
    existing = ['a', 'b']
    assert insert_tags(existing, ['x'], TagInsertMode.BEFORE_SELECTED,
                       selected_rows=[5, -1]) == ['a', 'b', 'x']


def test_empty_new_tags_is_noop():
    existing = ['a', 'b']
    for mode in TagInsertMode:
        assert insert_tags(existing, [], mode, 1, [0]) == ['a', 'b']


def test_inputs_not_mutated():
    existing = ['a', 'b']
    new_tags = ['x']
    insert_tags(existing, new_tags, TagInsertMode.REPLACE_SELECTED,
                selected_rows=[0])
    assert existing == ['a', 'b']
    assert new_tags == ['x']


def test_insert_into_empty_list():
    for mode in TagInsertMode:
        assert insert_tags([], ['x'], mode, 0, []) == ['x']


# ---------------------------------------------------------------------------
# Auto-caption "Insert at position"
# ---------------------------------------------------------------------------

def test_add_caption_insert_at_index(isolated_settings):
    from auto_captioning.captioning_thread import add_caption_to_tags
    from utils.enums import CaptionPosition
    tags = ['a', 'b', 'c']
    result = add_caption_to_tags(tags, 'x, y',
                                 CaptionPosition.INSERT_AT_INDEX, index=1)
    assert result == ['a', 'x', 'y', 'b', 'c']
    # Index 0 == top, large index clamps to bottom.
    assert add_caption_to_tags(tags, 'x', CaptionPosition.INSERT_AT_INDEX,
                               0) == ['x', 'a', 'b', 'c']
    assert add_caption_to_tags(tags, 'x', CaptionPosition.INSERT_AT_INDEX,
                               99) == ['a', 'b', 'c', 'x']


def test_add_caption_existing_positions_unchanged(isolated_settings):
    from auto_captioning.captioning_thread import add_caption_to_tags
    from utils.enums import CaptionPosition
    tags = ['a', 'b']
    assert add_caption_to_tags(tags, 'x', CaptionPosition.BEFORE_FIRST_TAG) == \
        ['x', 'a', 'b']
    assert add_caption_to_tags(tags, 'x', CaptionPosition.AFTER_LAST_TAG) == \
        ['a', 'b', 'x']
    assert add_caption_to_tags(tags, 'x', CaptionPosition.OVERWRITE_ALL_TAGS) == \
        ['x']
    assert add_caption_to_tags(tags, 'x', CaptionPosition.DO_NOT_ADD) == \
        ['a', 'b']


def test_caption_position_enum_has_insert_at_index():
    from utils.enums import CaptionPosition
    assert CaptionPosition.INSERT_AT_INDEX in list(CaptionPosition)


# ---------------------------------------------------------------------------
# Model-level multi-image add with insertion modes
# ---------------------------------------------------------------------------

def _model_with_images(tmp_path, tags_per_image):
    from pathlib import Path
    from models.image_list_model import ImageListModel
    from utils.image import Image
    model = ImageListModel(image_list_image_width=200, tag_separator=', ')
    images = []
    for i, tags in enumerate(tags_per_image):
        path = tmp_path / f'img{i}.png'
        path.write_bytes(b'')
        images.append(Image(path=path, dimensions=(10, 10), tags=list(tags)))
    model.images = images
    return model


def test_add_tags_at_index_across_images(tmp_path, isolated_settings):
    from utils.tag_insertion import TagInsertMode
    model = _model_with_images(tmp_path, [['a', 'b'], ['c']])
    indices = [model.index(0), model.index(1)]
    model.add_tags(['x'], indices, TagInsertMode.AT_INDEX, 1)
    assert model.images[0].tags == ['a', 'x', 'b']
    assert model.images[1].tags == ['c', 'x']  # index 1 clamps to bottom


def test_add_tags_top_across_images(tmp_path, isolated_settings):
    from utils.tag_insertion import TagInsertMode
    model = _model_with_images(tmp_path, [['a', 'b']])
    model.add_tags(['x'], [model.index(0)], TagInsertMode.TOP)
    assert model.images[0].tags == ['x', 'a', 'b']


def test_add_tags_selection_mode_falls_back_to_bottom(tmp_path,
                                                      isolated_settings):
    from utils.tag_insertion import TagInsertMode
    model = _model_with_images(tmp_path, [['a', 'b']])
    model.add_tags(['x'], [model.index(0)], TagInsertMode.BEFORE_SELECTED)
    assert model.images[0].tags == ['a', 'b', 'x']


def test_add_tags_default_appends(tmp_path, isolated_settings):
    model = _model_with_images(tmp_path, [['a']])
    model.add_tags(['x'], [model.index(0)])
    assert model.images[0].tags == ['a', 'x']
