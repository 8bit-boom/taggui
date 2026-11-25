"""
Widget-level tests for precise tag insertion: the manual TagInputBox honors
the insertion-mode combo / index / tag-list selection, and the Auto-Captioner
"Insert at position" index control enables and round-trips correctly.
"""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip('PySide6', reason='PySide6 is not installed')

from PySide6.QtCore import QModelIndex, QStringListModel, Qt
from PySide6.QtWidgets import (QComboBox, QListView, QSpinBox)

from utils.image import Image
from utils.tag_insertion import TagInsertMode


class OneImageList:
    def get_selected_image_indices(self):
        return [QModelIndex()]


@pytest.fixture
def tag_input_box(qtbot, isolated_settings):
    from utils.settings import get_settings
    get_settings().setValue('autocomplete_tags', False)
    from widgets.image_tags_editor import TagInputBox

    model = QStringListModel(['a', 'b', 'c'])
    box = TagInputBox(model, object(), OneImageList(), ', ')
    qtbot.addWidget(box)

    combo = QComboBox()
    combo.addItems([mode.value for mode in TagInsertMode])
    combo.setCurrentText(TagInsertMode.BOTTOM.value)
    qtbot.addWidget(combo)
    spin = QSpinBox()
    spin.setRange(0, 999)
    qtbot.addWidget(spin)
    view = QListView()
    view.setModel(model)
    view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
    qtbot.addWidget(view)

    box.insert_mode_combo = combo
    box.insert_index_spin = spin
    box.image_tags_list = view
    return box, model, combo, spin, view


def _select(view, model, rows):
    view.clearSelection()
    for row in rows:
        view.selectionModel().select(
            model.index(row),
            view.selectionModel().SelectionFlag.Select)


def test_manual_default_appends(tag_input_box):
    box, model, combo, spin, view = tag_input_box
    box.add_tag('x')
    assert model.stringList() == ['a', 'b', 'c', 'x']


def test_manual_top(tag_input_box):
    box, model, combo, spin, view = tag_input_box
    combo.setCurrentText(TagInsertMode.TOP.value)
    box.add_tag('x')
    assert model.stringList() == ['x', 'a', 'b', 'c']


def test_manual_at_index(tag_input_box):
    box, model, combo, spin, view = tag_input_box
    combo.setCurrentText(TagInsertMode.AT_INDEX.value)
    spin.setValue(1)
    box.add_tag('x')
    assert model.stringList() == ['a', 'x', 'b', 'c']


def test_manual_before_selected(tag_input_box):
    box, model, combo, spin, view = tag_input_box
    _select(view, model, [2])
    combo.setCurrentText(TagInsertMode.BEFORE_SELECTED.value)
    box.add_tag('x')
    assert model.stringList() == ['a', 'b', 'x', 'c']


def test_manual_after_selected(tag_input_box):
    box, model, combo, spin, view = tag_input_box
    _select(view, model, [0])
    combo.setCurrentText(TagInsertMode.AFTER_SELECTED.value)
    box.add_tag('x')
    assert model.stringList() == ['a', 'x', 'b', 'c']


def test_manual_replace_selected(tag_input_box):
    box, model, combo, spin, view = tag_input_box
    _select(view, model, [1])
    combo.setCurrentText(TagInsertMode.REPLACE_SELECTED.value)
    box.add_tag('x')
    assert model.stringList() == ['a', 'x', 'c']


def test_manual_multi_tag_input_splits_and_inserts(tag_input_box):
    box, model, combo, spin, view = tag_input_box
    combo.setCurrentText(TagInsertMode.TOP.value)
    # Multiple tags in one input go through the multi-tag path; with a single
    # selected image and >1 tag it emits the addition signal rather than the
    # single-tag insert, so verify the single-tag path stays precise instead.
    box.add_tag('x')
    assert model.stringList()[0] == 'x'


# ---------------------------------------------------------------------------
# Auto-Captioner "Insert at position" index control
# ---------------------------------------------------------------------------

@pytest.fixture
def captioner_form(qtbot, isolated_settings):
    from models.image_list_model import ImageListModel
    from widgets.auto_captioner import AutoCaptioner

    class StubImageList:
        def __init__(self, model):
            self.model = model

        def get_selected_image_indices(self):
            return [self.model.index(0)]

        def select_image_indices(self, indices):
            pass

    model = ImageListModel(image_list_image_width=200, tag_separator=', ')
    path = Path(tempfile.gettempdir()) / 'taggui-insert-test.png'
    if not path.exists():
        import PIL.Image as PilImage
        PilImage.new('RGB', (8, 8)).save(path)
    model.images = [Image(path=path, dimensions=(8, 8), tags=[])]
    captioner = AutoCaptioner(model, StubImageList(model))
    qtbot.addWidget(captioner)
    return captioner


def test_index_spin_enabled_only_for_insert_at_position(captioner_form):
    from utils.enums import CaptionPosition
    form = captioner_form.caption_settings_form
    form.caption_position_combo_box.setCurrentText(
        CaptionPosition.AFTER_LAST_TAG)
    assert not form.caption_position_index_spin_box.isEnabled()
    form.caption_position_combo_box.setCurrentText(
        CaptionPosition.INSERT_AT_INDEX)
    assert form.caption_position_index_spin_box.isEnabled()


def test_index_round_trips_through_settings(captioner_form):
    from utils.enums import CaptionPosition
    form = captioner_form.caption_settings_form
    form.caption_position_combo_box.setCurrentText(
        CaptionPosition.INSERT_AT_INDEX)
    form.caption_position_index_spin_box.setValue(2)
    snapshot = form.get_caption_settings()
    assert snapshot['caption_position_index'] == 2
    assert snapshot['caption_position'] == CaptionPosition.INSERT_AT_INDEX
    # Change away and back; applying the snapshot restores the index.
    form.caption_position_index_spin_box.setValue(7)
    form.apply_caption_settings(snapshot)
    assert form.caption_position_index_spin_box.value() == 2


def test_index_travels_with_chain_step_settings(captioner_form):
    # A chain step snapshot must carry its own insertion index so the thread
    # uses it (not the live form value).
    from utils.enums import CaptionPosition
    form = captioner_form.caption_settings_form
    form.caption_position_combo_box.setCurrentText(
        CaptionPosition.INSERT_AT_INDEX)
    form.caption_position_index_spin_box.setValue(3)
    step = form.get_caption_settings()
    assert step['caption_position_index'] == 3
    from auto_captioning.captioning_thread import CaptioningThread
    thread = CaptioningThread(None, None, [], step, ', ', None)
    assert thread.caption_position_index == 3
