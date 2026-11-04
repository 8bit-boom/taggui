try:
    # Only available in newer releases of `transformers`. Imported
    # defensively so that an older installed version doesn't prevent the
    # rest of the app (and every other model) from starting up; selecting
    # this model with an old `transformers` instead raises a clear error at
    # caption time via `get_additional_error_message()` below.
    from transformers import Qwen3_5MoeForCausalLM
except ImportError:
    Qwen3_5MoeForCausalLM = None

from auto_captioning.models.qwen3_5 import Qwen3_5


class Qwen3_5Moe(Qwen3_5):
    """
    The Mixture-of-Experts variant of Qwen3.5 (and later point releases
    sharing its architecture, such as Qwen3.6's "-A3B" checkpoints). Its
    `CausalLM` class has the same `pixel_values`-less `forward()` signature
    as the dense `Qwen3_5ForCausalLM`, so it reuses `Qwen3_5`'s
    `get_model_inputs()` image-embedding merge unchanged; only the
    `transformers` model class differs.
    """
    transformers_model_class = Qwen3_5MoeForCausalLM

    def get_additional_error_message(self) -> str | None:
        if Qwen3_5MoeForCausalLM is None:
            return ('Qwen3.5 MoE requires a newer version of the '
                    '`transformers` package than is currently installed. '
                    'Upgrade it with `pip install -U transformers`.')
        return None
