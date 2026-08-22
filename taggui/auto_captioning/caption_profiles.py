import json

from utils.settings import get_settings


def get_caption_profiles() -> dict[str, dict]:
    """Get the user's saved captioning profiles, keyed by name."""
    settings = get_settings()
    caption_profiles_json = settings.value('caption_profiles',
                                           defaultValue='{}', type=str)
    try:
        caption_profiles = json.loads(caption_profiles_json)
    except ValueError:
        caption_profiles = {}
    if not isinstance(caption_profiles, dict):
        caption_profiles = {}
    return caption_profiles


def save_caption_profiles(caption_profiles: dict[str, dict]):
    settings = get_settings()
    settings.setValue('caption_profiles', json.dumps(caption_profiles))
