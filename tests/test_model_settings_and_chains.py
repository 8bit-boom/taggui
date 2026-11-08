from auto_captioning.model_chains import get_model_chains, save_model_chains
from auto_captioning.model_settings import (get_model_settings,
                                             save_model_settings)


def test_model_settings_round_trip(isolated_settings):
    assert get_model_settings() == {}
    model_settings = {'Qwen/Qwen3-VL-8B-Instruct': {'prompt': 'Describe.'}}
    save_model_settings(model_settings)
    assert get_model_settings() == model_settings


def test_model_settings_default_on_corrupt_value(isolated_settings,
                                                  monkeypatch):
    from utils.settings import get_settings
    get_settings().setValue('model_settings', 'not valid json')
    assert get_model_settings() == {}
    get_settings().setValue('model_settings', '["a", "list", "not a dict"]')
    assert get_model_settings() == {}


def test_model_chains_round_trip(isolated_settings):
    assert get_model_chains() == {}
    chains = {
        'WD Tagger then Qwen': [
            {'model_id': 'SmilingWolf/wd-vit-tagger-v3',
             'caption_position': 'Overwrite'},
            {'model_id': 'Qwen/Qwen3-VL-8B-Instruct',
             'caption_position': 'Append to end',
             'prompt': 'Describe this image using the tags: {tags}'},
        ]
    }
    save_model_chains(chains)
    assert get_model_chains() == chains
    assert len(get_model_chains()['WD Tagger then Qwen']) == 2


def test_model_chains_default_on_corrupt_value(isolated_settings):
    from utils.settings import get_settings
    get_settings().setValue('model_chains', 'not valid json')
    assert get_model_chains() == {}
