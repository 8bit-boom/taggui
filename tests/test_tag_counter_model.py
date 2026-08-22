from pathlib import Path

from models.tag_counter_model import (DanbooruTagCompletionModel,
                                      TagCounterModel, load_danbooru_tags_csv)
from utils.image import Image


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding='utf-8')
    return path


def test_load_danbooru_tags_csv_with_header(tmp_path):
    csv_path = write_csv(
        tmp_path / 'tags.csv',
        'name,category,post_count,aliases\n'
        '1girl,0,1000000,\n'
        'solo,0,900000,\n'
        'rare_tag,0,5,\n')
    tags = load_danbooru_tags_csv(csv_path)
    assert tags == ['1girl', 'solo', 'rare_tag']


def test_load_danbooru_tags_csv_without_header(tmp_path):
    csv_path = write_csv(
        tmp_path / 'tags_no_header.csv',
        'rare_tag,0,5,\n'
        '1girl,0,1000000,\n'
        'solo,0,900000,\n')
    tags = load_danbooru_tags_csv(csv_path)
    assert tags == ['1girl', 'solo', 'rare_tag']


def test_load_danbooru_tags_csv_skips_unparseable_post_count_without_header(
        tmp_path):
    csv_path = write_csv(
        tmp_path / 'tags_bad.csv',
        'weird_row,0,notanumber,\n'
        '1girl,0,100,\n')
    tags = load_danbooru_tags_csv(csv_path)
    assert tags == ['1girl']


def test_danbooru_tag_completion_model_excludes_dataset_tags(qapp, tmp_path):
    csv_path = write_csv(
        tmp_path / 'tags.csv',
        'name,category,post_count,aliases\n'
        'solo,0,900000,\n'
        '1girl,0,1000000,\n'
        'outdoors,0,500000,\n')
    tag_counter_model = TagCounterModel()
    tag_counter_model.count_tags(
        [Image(path=Path('/a.jpg'), dimensions=(1, 1),
              tags=['1girl', 'smiling'])])
    completion_model = DanbooruTagCompletionModel(tag_counter_model)
    completion_model.load_danbooru_tags_csv(csv_path)
    # Dataset tags come first (in frequency order), then Danbooru tags with
    # already-present dataset tags excluded.
    assert completion_model.combined_tags == ['1girl', 'smiling', 'solo',
                                              'outdoors']


def test_danbooru_tag_completion_model_updates_on_dataset_change(qapp,
                                                                  tmp_path):
    csv_path = write_csv(tmp_path / 'tags.csv',
                         'name,category,post_count,aliases\nsolo,0,900000,\n')
    tag_counter_model = TagCounterModel()
    completion_model = DanbooruTagCompletionModel(tag_counter_model)
    completion_model.load_danbooru_tags_csv(csv_path)
    assert completion_model.combined_tags == ['solo']
    tag_counter_model.count_tags(
        [Image(path=Path('/a.jpg'), dimensions=(1, 1), tags=['solo'])])
    # `solo` is now a dataset tag, so it should no longer be duplicated from
    # the Danbooru list.
    assert completion_model.combined_tags == ['solo']
