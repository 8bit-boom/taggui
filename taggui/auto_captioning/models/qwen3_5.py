try:
    # Only available in newer releases of `transformers`. Imported
    # defensively so that an older installed version doesn't prevent the
    # rest of the app (and every other model) from starting up; selecting
    # this model with an old `transformers` instead raises a clear error at
    # caption time via `get_additional_error_message()` below.
    from transformers import Qwen3_5ForCausalLM
except ImportError:
    Qwen3_5ForCausalLM = None

from auto_captioning.models.qwen3_vl import Qwen3VL
from utils.image import Image


class Qwen3_5(Qwen3VL):
    """
    Qwen3.5 is natively multimodal: unlike Qwen3-VL, its `CausalLM` class does
    not accept `pixel_values` directly in `generate()`. The image has to be
    encoded with the model's visual encoder and merged into `inputs_embeds`
    before generation.
    """
    transformers_model_class = Qwen3_5ForCausalLM

    def get_additional_error_message(self) -> str | None:
        if Qwen3_5ForCausalLM is None:
            return ('Qwen3.5 requires a newer version of the `transformers` '
                    'package than is currently installed. Upgrade it with '
                    '`pip install -U transformers`.')
        return None

    def get_model_inputs(self, image_prompt: str, image: Image):
        text = self.get_input_text(image_prompt)
        pil_image = self.load_image(image)
        processor_inputs = (self.processor(text=text, images=pil_image,
                                           return_tensors='pt')
                            .to(self.device, **self.dtype_argument))
        input_ids = processor_inputs['input_ids']
        pixel_values = processor_inputs.pop('pixel_values', None)
        image_grid_thw = processor_inputs.pop('image_grid_thw', None)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.model.visual(pixel_values,
                                             grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.dtype)
            image_token_id = self.model.config.image_token_id
            image_token_mask = input_ids == image_token_id
            inputs_embeds[image_token_mask] = image_embeds
        model_inputs = dict(processor_inputs)
        model_inputs['inputs_embeds'] = inputs_embeds
        del model_inputs['input_ids']
        return model_inputs
