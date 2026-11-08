import json

from utils.settings import get_settings


def get_model_chains() -> dict[str, list[dict]]:
    """
    Get the user's saved model chains, keyed by name. Each chain is an
    ordered list of full captioning-settings snapshots (the same shape as
    `CaptionSettingsForm.get_caption_settings()`), one per step.
    """
    settings = get_settings()
    model_chains_json = settings.value('model_chains', defaultValue='{}',
                                       type=str)
    try:
        model_chains = json.loads(model_chains_json)
    except ValueError:
        model_chains = {}
    if not isinstance(model_chains, dict):
        model_chains = {}
    return model_chains


def save_model_chains(model_chains: dict[str, list[dict]]):
    settings = get_settings()
    settings.setValue('model_chains', json.dumps(model_chains))
