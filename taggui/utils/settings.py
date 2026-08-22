import json

from PySide6.QtCore import QSettings

# Defaults for settings that are accessed from multiple places.
DEFAULT_SETTINGS = {
    'font_size': 16,
    # Common image formats that are supported in PySide6.
    'image_list_file_formats':
        'avif, bmp, gif, jpg, jpeg, png, tif, tiff, webp',
    'image_list_image_width': 200,
    'tag_separator': ',',
    'insert_space_after_tag_separator': True,
    'autocomplete_tags': True,
    'models_directory_path': '',
    'danbooru_tags_csv_path': ''
}


def get_settings() -> QSettings:
    settings = QSettings('taggui', 'taggui')
    return settings


def get_tag_separator() -> str:
    settings = get_settings()
    tag_separator = settings.value(
        'tag_separator', defaultValue=DEFAULT_SETTINGS['tag_separator'],
        type=str)
    insert_space_after_tag_separator = settings.value(
        'insert_space_after_tag_separator',
        defaultValue=DEFAULT_SETTINGS['insert_space_after_tag_separator'],
        type=bool)
    if insert_space_after_tag_separator:
        tag_separator += ' '
    return tag_separator


def get_saved_filters() -> dict[str, str]:
    """Get the user's saved image list filters, keyed by name."""
    settings = get_settings()
    saved_filters_json = settings.value('saved_filters', defaultValue='{}',
                                        type=str)
    try:
        saved_filters = json.loads(saved_filters_json)
    except ValueError:
        saved_filters = {}
    if not isinstance(saved_filters, dict):
        saved_filters = {}
    return saved_filters


def save_saved_filters(saved_filters: dict[str, str]):
    settings = get_settings()
    settings.setValue('saved_filters', json.dumps(saved_filters))


def get_tag_presets() -> list[dict]:
    """Get the user's tag presets: `[{'name': str, 'tags': list[str]}, ...]`."""
    settings = get_settings()
    tag_presets_json = settings.value('tag_presets', defaultValue='[]',
                                      type=str)
    try:
        tag_presets = json.loads(tag_presets_json)
    except ValueError:
        tag_presets = []
    if not isinstance(tag_presets, list):
        tag_presets = []
    return tag_presets


def save_tag_presets(tag_presets: list[dict]):
    settings = get_settings()
    settings.setValue('tag_presets', json.dumps(tag_presets))
