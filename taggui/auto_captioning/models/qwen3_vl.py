import re

import torch
from transformers import AutoProcessor

try:
    # Only available in newer releases of `transformers`. Imported
    # defensively so that an older installed version doesn't prevent the
    # rest of the app (and every other model) from starting up; selecting
    # this model with an old `transformers` instead raises a clear error at
    # caption time via `get_additional_error_message()` below.
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

# Tuned to keep VRAM usage reasonable on a 32 GB GPU.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

THINK_BLOCK_PATTERN = re.compile(r'(?:<think>)?.*?</think>\s*', re.DOTALL)


class Qwen3VL(AutoCaptioningModel):
    dtype = torch.bfloat16
    transformers_model_class = Qwen3VLForConditionalGeneration

    def get_additional_error_message(self) -> str | None:
        if Qwen3VLForConditionalGeneration is None:
            return ('Qwen3-VL requires a newer version of the `transformers` '
                    'package than is currently installed. Upgrade it with '
                    '`pip install -U transformers`.')
        return None

    @staticmethod
    def get_quantization_skip_modules() -> list[str] | None:
        # Matches the `llm_int8_skip_modules` used by community-published
        # 4-bit/8-bit Qwen3-VL checkpoints: the vision tower (`visual`) and
        # its patch merger (`merger`) live under `model.visual` in this
        # model family, and `lm_head` is a top-level attribute - both are
        # matched as substrings of each module's full dotted name.
        return ['visual', 'merger', 'lm_head']

    def get_processor(self):
        return AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS)

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in detail.'

    def format_prompt(self, prompt: str) -> str:
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt}
                ]
            }
        ]
        templated_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            # Qwen3.5/3.6's chat template puts the opening `<think>` tag in
            # the prompt itself (not the generated output), so
            # `enable_thinking=False` is what actually turns reasoning off
            # for hybrid think/no-think models; models that don't define
            # this template variable just ignore the extra argument. A
            # dedicated "-Thinking"-only checkpoint may not honor this at
            # all (it isn't given a real choice between modes).
            enable_thinking=not self.disable_thinking)
        return templated_prompt

    def get_input_text(self, image_prompt: str) -> str:
        return image_prompt + self.caption_start

    def get_model_inputs(self, image_prompt: str, image: Image):
        text = self.get_input_text(image_prompt)
        pil_image = self.load_image(image)
        model_inputs = (self.processor(text=text, images=pil_image,
                                       return_tensors='pt')
                        .to(self.device, **self.dtype_argument))
        # Stashed so `get_caption_from_generated_tokens` below can slice off
        # the input tokens before decoding, instead of string-matching the
        # templated prompt against the decoded text. The latter requires
        # knowing every special token's literal text to strip (so it lines
        # up with a `skip_special_tokens=True` decode), which is fragile:
        # Qwen3.5's tokenizer uses a completely different special-token ID
        # range than Qwen3-VL's (e.g. its image token is 248056, not
        # Qwen3-VL's 151655), silently breaking a hardcoded token list
        # written against one of them. Slicing by input length doesn't
        # depend on knowing any of that.
        self.input_token_count = model_inputs['input_ids'].shape[-1]
        return model_inputs

    def get_caption_from_generated_tokens(
            self, generated_token_ids: torch.Tensor, image_prompt: str) -> str:
        generated_token_ids = generated_token_ids[:, self.input_token_count:]
        return super().get_caption_from_generated_tokens(generated_token_ids,
                                                          image_prompt)

    @staticmethod
    def postprocess_generated_text(generated_text: str) -> str:
        # Qwen3-VL's reasoning models may emit a `<think>...</think>` block
        # before the actual caption. Strip it out. The opening `<think>` is
        # usually forced by the chat template as part of the generation
        # prompt rather than actually generated, so by the time this runs
        # `generated_text` (already sliced to just the newly generated
        # tokens in `get_caption_from_generated_tokens` above) often starts
        # mid-reasoning with no opening tag at all - only the closing
        # `</think>`. The pattern below matches that case too, not just a
        # self-contained `<think>...</think>` pair.
        return THINK_BLOCK_PATTERN.sub('', generated_text).strip()
