import re

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

# Tuned to keep VRAM usage reasonable on a 32 GB GPU.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

THINK_BLOCK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)


class Qwen3VL(AutoCaptioningModel):
    dtype = torch.bfloat16
    transformers_model_class = Qwen3VLForConditionalGeneration

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
            conversation, tokenize=False, add_generation_prompt=True)
        return templated_prompt

    def get_input_text(self, image_prompt: str) -> str:
        return image_prompt + self.caption_start

    def get_model_inputs(self, image_prompt: str, image: Image):
        text = self.get_input_text(image_prompt)
        pil_image = self.load_image(image)
        model_inputs = (self.processor(text=text, images=pil_image,
                                       return_tensors='pt')
                        .to(self.device, **self.dtype_argument))
        return model_inputs

    @staticmethod
    def postprocess_generated_text(generated_text: str) -> str:
        # Qwen3-VL's reasoning models may emit a `<think>...</think>` block
        # before the actual caption. Strip it out.
        return THINK_BLOCK_PATTERN.sub('', generated_text).strip()
