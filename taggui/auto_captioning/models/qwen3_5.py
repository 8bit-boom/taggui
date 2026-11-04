try:
    # Only available in newer releases of `transformers`. Imported
    # defensively so that an older installed version doesn't prevent the
    # rest of the app (and every other model) from starting up; selecting
    # this model with an old `transformers` instead raises a clear error at
    # caption time via `get_additional_error_message()` below.
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

from auto_captioning.models.qwen3_vl import Qwen3VL


class Qwen3_5(Qwen3VL):
    """
    `Qwen3_5ForConditionalGeneration` accepts `pixel_values` directly in
    `generate()` (merging the image embeddings internally), the same as
    `Qwen3VLForConditionalGeneration`, so no extra input handling is needed
    here - only the model class differs from `Qwen3VL`.

    (An earlier `transformers` release had `Qwen3_5ForCausalLM` double as
    the multimodal entry point, requiring the image to be manually encoded
    and merged into `inputs_embeds` before generation, since `generate()`
    didn't accept `pixel_values` directly. Current releases turned
    `Qwen3_5ForCausalLM` into a text-only class with no visual encoder at
    all - selecting it now raises `AttributeError: 'Qwen3_5ForCausalLM'
    object has no attribute 'visual'` - and moved multimodal support to
    `Qwen3_5ForConditionalGeneration` instead.)
    """
    transformers_model_class = Qwen3_5ForConditionalGeneration

    def get_additional_error_message(self) -> str | None:
        if Qwen3_5ForConditionalGeneration is None:
            return ('Qwen3.5 requires a newer version of the `transformers` '
                    'package than is currently installed. Upgrade it with '
                    '`pip install -U transformers`.')
        return None
