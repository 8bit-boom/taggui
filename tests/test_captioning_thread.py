from auto_captioning.captioning_thread import add_caption_to_tags, format_duration
from utils.enums import CaptionPosition

# `add_caption_to_tags` splits the caption using the default tag separator
# (`, `), so tags in the caption must be separated with a space after the
# comma to be split apart correctly.
CAPTION = 'c, d'


def test_add_caption_to_tags_do_not_add(isolated_settings):
    tags = ['a', 'b']
    result = add_caption_to_tags(tags, CAPTION, CaptionPosition.DO_NOT_ADD)
    assert result == ['a', 'b']


def test_add_caption_to_tags_empty_caption(isolated_settings):
    tags = ['a', 'b']
    result = add_caption_to_tags(tags, '', CaptionPosition.BEFORE_FIRST_TAG)
    assert result == ['a', 'b']


def test_add_caption_to_tags_before_first_tag(isolated_settings):
    tags = ['a', 'b']
    result = add_caption_to_tags(tags, CAPTION,
                                 CaptionPosition.BEFORE_FIRST_TAG)
    assert result == ['c', 'd', 'a', 'b']


def test_add_caption_to_tags_after_last_tag(isolated_settings):
    tags = ['a', 'b']
    result = add_caption_to_tags(tags, CAPTION, CaptionPosition.AFTER_LAST_TAG)
    assert result == ['a', 'b', 'c', 'd']


def test_add_caption_to_tags_overwrite_first_tag(isolated_settings):
    tags = ['a', 'b']
    result = add_caption_to_tags(tags, CAPTION,
                                 CaptionPosition.OVERWRITE_FIRST_TAG)
    assert result == ['c', 'd', 'b']


def test_add_caption_to_tags_overwrite_first_tag_when_no_tags(
        isolated_settings):
    result = add_caption_to_tags([], CAPTION,
                                 CaptionPosition.OVERWRITE_FIRST_TAG)
    assert result == ['c', 'd']


def test_add_caption_to_tags_overwrite_all_tags(isolated_settings):
    tags = ['a', 'b']
    result = add_caption_to_tags(tags, CAPTION,
                                 CaptionPosition.OVERWRITE_ALL_TAGS)
    assert result == ['c', 'd']


def test_add_caption_to_tags_does_not_mutate_original(isolated_settings):
    tags = ['a', 'b']
    add_caption_to_tags(tags, CAPTION, CaptionPosition.BEFORE_FIRST_TAG)
    assert tags == ['a', 'b']


def test_format_duration_seconds():
    assert format_duration(5.4) == '5.4 seconds'


def test_format_duration_minutes():
    assert format_duration(90) == '1.5 minutes'


def test_format_duration_hours():
    assert format_duration(3600 * 2.5) == '2.5 hours'


def test_format_duration_days():
    assert format_duration(3600 * 24 * 1.5) == '1.5 days'
