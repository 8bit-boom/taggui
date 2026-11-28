"""
Automated tests for the captioning configuration system: generation
presets, load presets, prompt templates ("command prompts"), captioner
suffixes, caption profiles, and how they line up with model routing.

These are data-integrity + behavior tests: they catch typos, missing
matrix entries, out-of-range values, and drift between the preset data
and the widget that consumes it - the kinds of regressions that silently
produce a broken or empty caption for a whole model family.
"""

import pytest

from auto_captioning.generation_presets import GENERATION_PRESETS
from auto_captioning.load_presets import LOAD_PRESETS
from auto_captioning.prompt_templates import (
    CAPTIONER_PROMPT_SUFFIXES,
    PROMPT_TEMPLATES,
    PROMPT_TEMPLATE_CAPTIONERS,
    PROMPT_TEMPLATE_PURPOSES,
)

# The exact keys the Auto-Captioner widget reads/writes for generation
# parameters (`CaptionSettingsForm.get_caption_settings`). A preset must
# provide precisely these so `apply_generation_preset` can set every
# widget without a KeyError.
GENERATION_PARAMETER_KEYS = {
    'min_new_tokens', 'max_new_tokens', 'num_beams', 'length_penalty',
    'do_sample', 'temperature', 'top_k', 'top_p', 'repetition_penalty',
    'no_repeat_ngram_size',
}

# Ranges enforced by the widget's spin boxes; a preset outside these would
# be silently clamped when applied, so the data should already be in range.
GENERATION_RANGES = {
    'min_new_tokens': (1, 999),
    'max_new_tokens': (1, 999),
    'num_beams': (1, 99),
    'length_penalty': (-5, 5),
    'temperature': (0.01, 2),
    'top_k': (0, 200),
    'top_p': (0, 1),
    'repetition_penalty': (1, 2),
    'no_repeat_ngram_size': (0, 5),
}

# Settings keys a load preset is allowed to touch (subset of the widget's
# settings). Anything else would be a typo that silently does nothing.
LOAD_PRESET_KEYS = {
    'load_in_4_bit', 'load_in_8_bit', 'cpu_offload', 'sage_attention',
    'disable_thinking', 'attention_implementation', 'torch_compile',
    'quantize_kv_cache', 'batch_size', 'min_image_megapixels',
    'max_image_megapixels', 'wd_tagger_load_in_8_bit',
}

VALID_ATTENTION_IMPLEMENTATIONS = {'Default', 'SDPA', 'FlashAttention 2'}

# Models whose checkpoints reason before answering (a `</arg_value>` block),
# so they need a larger token budget and must not n-gram-block the
# legitimately repetitive reasoning text.
REASONING_CAPTIONERS = {'Qwen3-VL', 'Qwen3.5'}


# ---------------------------------------------------------------------------
# Generation presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('captioner', sorted(GENERATION_PRESETS))
def test_generation_preset_has_exact_widget_keys(captioner):
    preset = GENERATION_PRESETS[captioner]
    assert set(preset) == GENERATION_PARAMETER_KEYS, (
        f'{captioner} preset keys differ from the widget generation '
        f'parameters; apply_generation_preset would KeyError or leave a '
        f'widget unset.')


@pytest.mark.parametrize('captioner', sorted(GENERATION_PRESETS))
def test_generation_preset_values_in_widget_range(captioner):
    preset = GENERATION_PRESETS[captioner]
    for key, (low, high) in GENERATION_RANGES.items():
        value = preset[key]
        assert low <= value <= high, (
            f'{captioner}.{key}={value} outside widget range [{low}, {high}]')
    assert isinstance(preset['do_sample'], bool)
    assert preset['min_new_tokens'] <= preset['max_new_tokens'], (
        f'{captioner}: min_new_tokens exceeds max_new_tokens')


@pytest.mark.parametrize('captioner', sorted(REASONING_CAPTIONERS))
def test_reasoning_presets_leave_room_for_think_block(captioner):
    preset = GENERATION_PRESETS[captioner]
    direct = GENERATION_PRESETS['JoyCaption']
    # A reasoning model spends most of its budget on the `</arg_value>` block
    # before the caption; it must have a substantially larger budget than a
    # direct captioner or the caption gets truncated mid-reasoning.
    assert preset['max_new_tokens'] > direct['max_new_tokens'], (
        f'{captioner} needs a larger token budget than JoyCaption')
    # Reasoning text legitimately repeats phrases; an n-gram block would
    # mangle it, so it must be disabled.
    assert preset['no_repeat_ngram_size'] == 0


def test_joycaption_preset_curbs_repetition():
    preset = GENERATION_PRESETS['JoyCaption']
    # A direct captioner over a full paragraph tends to repeat phrases, so
    # it enables an n-gram block and a repetition penalty > 1.
    assert preset['no_repeat_ngram_size'] > 0
    assert preset['repetition_penalty'] > 1.0


def test_qwen_presets_match_published_sampling_recommendation():
    # Qwen's published Qwen3 recommendation: temperature 0.7, top_p 0.8,
    # top_k 20. Both Qwen presets share it.
    for captioner in REASONING_CAPTIONERS:
        preset = GENERATION_PRESETS[captioner]
        assert preset['temperature'] == pytest.approx(0.7)
        assert preset['top_p'] == pytest.approx(0.8)
        assert preset['top_k'] == 20
        assert preset['do_sample'] is True


def test_generation_presets_cover_the_prompt_captioners():
    # Every selectable captioner suffix (other than "(none)") should have a
    # matching generation preset, so picking a captioner always has sensible
    # sampling settings available.
    selectable = set(PROMPT_TEMPLATE_CAPTIONERS) - {'(none)'}
    assert selectable <= set(GENERATION_PRESETS), (
        f'captioners without a generation preset: '
        f'{selectable - set(GENERATION_PRESETS)}')


# ---------------------------------------------------------------------------
# Load presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(LOAD_PRESETS))
def test_load_preset_keys_are_known_settings(name):
    preset = LOAD_PRESETS[name]
    assert preset, f'{name} preset is empty'
    unknown = set(preset) - LOAD_PRESET_KEYS
    assert not unknown, f'{name} preset has unknown keys: {unknown}'


@pytest.mark.parametrize('name', sorted(LOAD_PRESETS))
def test_load_preset_values_are_sane(name):
    preset = LOAD_PRESETS[name]
    if 'load_in_4_bit' in preset and 'load_in_8_bit' in preset:
        assert not (preset['load_in_4_bit'] and preset['load_in_8_bit']), (
            f'{name}: 4-bit and 8-bit are mutually exclusive')
    if 'batch_size' in preset:
        assert 1 <= preset['batch_size'] <= 64
    if 'attention_implementation' in preset:
        assert preset['attention_implementation'] in \
            VALID_ATTENTION_IMPLEMENTATIONS
    if 'min_image_megapixels' in preset and 'max_image_megapixels' in preset:
        assert preset['min_image_megapixels'] <= preset['max_image_megapixels']
    for key in ('load_in_4_bit', 'load_in_8_bit', 'cpu_offload',
                'sage_attention', 'disable_thinking', 'torch_compile',
                'quantize_kv_cache', 'wd_tagger_load_in_8_bit'):
        if key in preset:
            assert isinstance(preset[key], bool), f'{name}.{key} not bool'


def test_wd_tagger_load_preset_targets_the_tagger():
    preset = LOAD_PRESETS['WD Tagger']
    assert preset.get('wd_tagger_load_in_8_bit') is True
    # The WD tagger is ONNX, not torch - it must not request torch.compile.
    assert preset.get('torch_compile', False) is False


def test_moe_load_preset_quantizes_and_uses_fp8_kv_cache():
    preset = LOAD_PRESETS['Qwen3.5 MoE (A3B)']
    assert preset['load_in_4_bit'] is True
    assert preset['quantize_kv_cache'] is True


# ---------------------------------------------------------------------------
# Prompt templates ("command prompts")
# ---------------------------------------------------------------------------

def test_every_model_defines_every_purpose():
    # A complete model x purpose matrix; a missing entry would KeyError in
    # insert_prompt_template when the user picks that combination.
    for model, purposes in PROMPT_TEMPLATES.items():
        assert set(purposes) == set(PROMPT_TEMPLATE_PURPOSES), (
            f'{model} purposes {set(purposes)} != {set(PROMPT_TEMPLATE_PURPOSES)}')


@pytest.mark.parametrize('model', sorted(PROMPT_TEMPLATES))
@pytest.mark.parametrize('purpose', PROMPT_TEMPLATE_PURPOSES)
def test_prompt_template_is_nonempty_text(model, purpose):
    text = PROMPT_TEMPLATES[model][purpose]
    assert isinstance(text, str)
    assert text.strip(), f'{model}/{purpose} prompt is empty'
    # A command prompt should be an instruction, not a stray fragment.
    assert len(text.split()) >= 5, f'{model}/{purpose} prompt too short'


def test_natural_language_style_templates_omit_style_words():
    # Documented convention: for NL-caption models a Style caption describes
    # content only, so the style is learned from the images, not the text.
    nl_models = ['SDXL', 'Flux.1', 'Flux.2', 'Z-Image', 'Krea 2 Turbo']
    for model in nl_models:
        style = PROMPT_TEMPLATES[model]['Style'].lower()
        assert 'do not mention' in style, (
            f'{model} Style template should instruct to omit style words')
        assert 'style' in style


def test_anima_style_template_asks_for_style_tags():
    # Anima is the exception: Danbooru treats medium/technique/artist tags as
    # normal controllable tags, so its Style template asks for them.
    style = PROMPT_TEMPLATES['Anima']['Style'].lower()
    assert 'tag' in style
    assert 'style' in style


def test_anima_templates_use_danbooru_tags():
    for purpose in PROMPT_TEMPLATE_PURPOSES:
        text = PROMPT_TEMPLATES['Anima'][purpose].lower()
        assert 'danbooru' in text or 'tag' in text, (
            f'Anima/{purpose} should request Danbooru-style tags')


def test_sdxl_templates_stay_short_for_clip_token_limit():
    # SDXL's CLIP encoders truncate ~75 tokens; its General template must
    # ask for a short caption (a single sentence / word cap).
    general = PROMPT_TEMPLATES['SDXL']['General'].lower()
    assert 'sentence' in general
    assert 'word' in general or 'token' in general


@pytest.mark.parametrize('model', ['Flux.2', 'Z-Image', 'Krea 2 Turbo'])
def test_long_context_character_templates_capture_face_detail(model):
    # These targets use long-context encoders; their Character templates
    # catalog full appearance (hair/eyes/face) rather than withholding it.
    character = PROMPT_TEMPLATES[model]['Character'].lower()
    assert 'hair' in character
    assert 'eyes' in character or 'eye' in character


# ---------------------------------------------------------------------------
# Captioner suffixes + combination logic
# ---------------------------------------------------------------------------

def test_captioner_suffix_keys_are_selectable_captioners():
    selectable = set(PROMPT_TEMPLATE_CAPTIONERS) - {'(none)'}
    assert set(CAPTIONER_PROMPT_SUFFIXES) == selectable


@pytest.mark.parametrize('captioner', ['Qwen3-VL', 'Qwen3.5'])
def test_qwen_suffix_suppresses_reasoning_and_preamble(captioner):
    suffix = CAPTIONER_PROMPT_SUFFIXES[captioner].lower()
    assert 'think' in suffix
    assert 'preamble' in suffix or 'reasoning' in suffix


def test_joycaption_suffix_suppresses_moralizing():
    suffix = CAPTIONER_PROMPT_SUFFIXES['JoyCaption'].lower()
    assert 'moralize' in suffix or 'refuse' in suffix


def combine_template(model, purpose, captioner):
    """Mirror `CaptionSettingsForm.insert_prompt_template` exactly."""
    template = PROMPT_TEMPLATES[model][purpose]
    suffix = CAPTIONER_PROMPT_SUFFIXES.get(captioner)
    if suffix:
        template = f'{template} {suffix}'
    return template


@pytest.mark.parametrize('model', sorted(PROMPT_TEMPLATES))
@pytest.mark.parametrize('captioner', PROMPT_TEMPLATE_CAPTIONERS)
def test_template_suffix_combination_for_all_models(model, captioner):
    # Every model x captioner combination must produce a usable prompt; this
    # is the full "command prompt" surface the Insert button can generate.
    combined = combine_template(model, 'General', captioner)
    base = PROMPT_TEMPLATES[model]['General']
    assert combined.startswith(base)
    if captioner == '(none)':
        assert combined == base
    else:
        suffix = CAPTIONER_PROMPT_SUFFIXES[captioner]
        assert combined == f'{base} {suffix}'
        assert combined.endswith(suffix)


def test_none_captioner_adds_no_suffix():
    assert CAPTIONER_PROMPT_SUFFIXES.get('(none)') is None
    for model in PROMPT_TEMPLATES:
        assert combine_template(model, 'General', '(none)') == \
            PROMPT_TEMPLATES[model]['General']


# ---------------------------------------------------------------------------
# Caption profiles (persistence)
# ---------------------------------------------------------------------------

def test_caption_profiles_round_trip(isolated_settings):
    from auto_captioning.caption_profiles import (
        get_caption_profiles, save_caption_profiles)
    assert get_caption_profiles() == {}
    profiles = {
        'qwen-style': {
            'model_id': 'Qwen/Qwen3-VL-8B-Instruct',
            'prompt': combine_template('Flux.2', 'Style', 'Qwen3-VL'),
            'generation_parameters': dict(GENERATION_PRESETS['Qwen3-VL']),
        },
    }
    save_caption_profiles(profiles)
    assert get_caption_profiles() == profiles


def test_caption_profiles_corrupt_json_is_empty(isolated_settings):
    from auto_captioning.caption_profiles import get_caption_profiles
    from utils.settings import get_settings
    get_settings().setValue('caption_profiles', 'not valid json')
    assert get_caption_profiles() == {}


def test_caption_profiles_non_dict_is_empty(isolated_settings):
    from auto_captioning.caption_profiles import get_caption_profiles
    from utils.settings import get_settings
    get_settings().setValue('caption_profiles', '[1, 2, 3]')
    assert get_caption_profiles() == {}


# ---------------------------------------------------------------------------
# Model routing <-> preset consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model_id,expected_preset', [
    ('Qwen/Qwen3-VL-8B-Instruct', 'Qwen3-VL'),
    ('huihui-ai/Qwen3.5-8B-Instruct-abliterated', 'Qwen3.5'),
    ('fancyfeast/llama-joycaption-beta-one-hf-llava', 'JoyCaption'),
])
def test_model_routes_to_a_family_with_a_generation_preset(
        model_id, expected_preset):
    from auto_captioning.models_list import get_model_class
    model_class = get_model_class(model_id)
    assert model_class is not None
    assert expected_preset in GENERATION_PRESETS


def test_qwen_moe_routes_to_moe_class():
    from auto_captioning.models_list import get_model_class
    from auto_captioning.models.qwen3_5_moe import Qwen3_5Moe
    assert get_model_class('Qwen/Qwen3.6-35B-A3B') is Qwen3_5Moe


def test_wd_tagger_opts_out_of_torch_compile():
    from auto_captioning.models_list import get_model_class
    model_class = get_model_class('SmilingWolf/wd-eva02-large-tagger-v3')
    assert model_class.supports_torch_compile is False
