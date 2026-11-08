import json

from utils.settings import get_settings


def get_model_settings() -> dict[str, dict]:
    """Get the user's saved per-model settings, keyed by model ID."""
    settings = get_settings()
    model_settings_json = settings.value('model_settings',
                                         defaultValue='{}', type=str)
    try:
        model_settings = json.loads(model_settings_json)
    except ValueError:
        model_settings = {}
    if not isinstance(model_settings, dict):
        model_settings = {}
    return model_settings


def save_model_settings(model_settings: dict[str, dict]):
    settings = get_settings()
    settings.setValue('model_settings', json.dumps(model_settings))
