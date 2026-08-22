from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning.models.florence_2 import Florence2, Florence2Promptgen
from auto_captioning.models.joycaption import Joycaption
from auto_captioning.models.kosmos_2 import Kosmos2
from auto_captioning.models.llava_1_point_5 import Llava1Point5
from auto_captioning.models.moondream import Moondream1, Moondream2
from auto_captioning.models.phi_3_vision import Phi3Vision
from auto_captioning.models.qwen3_5 import Qwen3_5
from auto_captioning.models.qwen3_vl import Qwen3VL
from auto_captioning.models.wd_tagger import WdTagger
from auto_captioning.models_list import (get_all_models, get_model_class,
                                         get_user_models, save_user_models)


def test_get_model_class_defaults_to_base_class():
    assert get_model_class('some/unknown-model') is AutoCaptioningModel


def test_get_model_class_florence():
    assert get_model_class('microsoft/Florence-2-large') is Florence2


def test_get_model_class_florence_promptgen():
    assert (get_model_class('MiaoshouAI/Florence-2-large-PromptGen-v2.0')
           is Florence2Promptgen)


def test_get_model_class_joycaption():
    assert (get_model_class('fancyfeast/llama-joycaption-beta-one-hf-llava')
           is Joycaption)


def test_get_model_class_kosmos():
    assert get_model_class('microsoft/kosmos-2-patch14-224') is Kosmos2


def test_get_model_class_llava_1_5():
    assert get_model_class('llava-hf/llava-1.5-7b-hf') is Llava1Point5


def test_get_model_class_moondream():
    assert get_model_class('vikhyatk/moondream1') is Moondream1
    assert get_model_class('vikhyatk/moondream2') is Moondream2


def test_get_model_class_phi_3_vision():
    assert (get_model_class('microsoft/Phi-3-vision-128k-instruct')
           is Phi3Vision)


def test_get_model_class_wd_tagger():
    assert get_model_class('SmilingWolf/wd-vit-large-tagger-v3') is WdTagger


def test_get_model_class_qwen3_vl():
    assert get_model_class('Qwen/Qwen3-VL-8B-Instruct') is Qwen3VL
    assert (get_model_class('huihui-ai/Qwen3-VL-8B-Instruct-abliterated')
           is Qwen3VL)


def test_get_model_class_qwen3_5():
    assert (get_model_class('huihui-ai/Qwen3.5-8B-Instruct-abliterated')
           is Qwen3_5)


def test_get_user_models_round_trip(isolated_settings):
    assert get_user_models() == []
    save_user_models(['my-org/my-model', '/local/path/to/model'])
    assert get_user_models() == ['my-org/my-model', '/local/path/to/model']


def test_get_user_models_defaults_to_empty_list_when_corrupt(isolated_settings):
    from utils.settings import get_settings
    settings = get_settings()
    settings.setValue('user_models', 'not valid json')
    assert get_user_models() == []


def test_get_all_models_prepends_user_models(isolated_settings):
    save_user_models(['my-org/my-model'])
    all_models = get_all_models()
    assert all_models[0] == 'my-org/my-model'
    assert 'microsoft/Florence-2-large-ft' in all_models
