"""
Generation-parameter presets for common captioning models.

Different captioning models respond very differently to the same sampling
settings. A reasoning model like Qwen3-VL/Qwen3.5 writes a `<think>...`
block before its actual answer, which can easily run past a low
`max_new_tokens` and get cut off mid-reasoning with no caption at all - it
needs a much larger token budget than a direct captioner like JoyCaption,
which answers immediately and rarely needs more than a couple hundred
tokens for a full paragraph.

`GENERATION_PRESETS[captioner]` returns a dict with the same keys as
`CaptionSettingsForm.get_caption_settings()['generation_parameters']`, so
it can be applied with the same widget-setting calls used for Captioning
Profiles.
"""

# Qwen's own published recommendation for Qwen3 (temperature 0.7, top_p 0.8,
# top_k 20) - min/max tokens and no_repeat_ngram_size tuned here for
# captioning specifically: a generous max_new_tokens leaves room for the
# <think> block *and* the actual caption after it without truncating
# mid-reasoning, and no_repeat_ngram_size is left at 0 (disabled) since
# reasoning text legitimately repeats phrases (e.g. re-stating what it's
# looking at) that a strict n-gram block would otherwise mangle.
_QWEN_PRESET = {
    'min_new_tokens': 1,
    'max_new_tokens': 512,
    'num_beams': 1,
    'length_penalty': 1.0,
    'do_sample': True,
    'temperature': 0.7,
    'top_k': 20,
    'top_p': 0.8,
    'repetition_penalty': 1.0,
    'no_repeat_ngram_size': 0,
}

GENERATION_PRESETS: dict[str, dict] = {
    'Qwen3-VL': _QWEN_PRESET,
    'Qwen3.5': _QWEN_PRESET,
    # JoyCaption answers directly with no reasoning step, so it needs a
    # much smaller token budget than the Qwen presets above. These are
    # general-purpose captioning settings (not a vendor-published
    # recommendation like Qwen's), tuned to curb the repeated phrases VLM
    # captioners commonly fall into over a full paragraph.
    'JoyCaption': {
        'min_new_tokens': 1,
        'max_new_tokens': 300,
        'num_beams': 1,
        'length_penalty': 1.0,
        'do_sample': True,
        'temperature': 0.6,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
        'no_repeat_ngram_size': 3,
    },
}
