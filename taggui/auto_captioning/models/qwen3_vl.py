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

from auto_captioning.auto_captioning_model import (AutoCaptioningModel,
                                                   THINK_BLOCK_PATTERN)
from utils.image import Image

# Tuned to keep VRAM usage reasonable on a 32 GB GPU. The UI exposes both
# bounds as megapixel settings (`min_image_megapixels`/
# `max_image_megapixels`); these constants are their defaults.
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28
# Every 28x28-pixel patch (after the 2x2 merge) becomes one visual token,
# so the pixel budget directly sets the visual token count - and with it
# the prefill cost and the VRAM needed for a given batch size.
PIXELS_PER_VISUAL_TOKEN = 28 * 28


class Qwen3VL(AutoCaptioningModel):
    dtype = torch.bfloat16
    transformers_model_class = Qwen3VLForConditionalGeneration
    # Qwen-style processors accept a list of texts and images and pad them
    # into one batch, so several images can be captioned per `generate()`
    # call (the biggest in-process speedup).
    supports_batching = True

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

    def get_pixels(self) -> tuple[int, int]:
        megapixels_to_pixels = 1_000_000
        min_megapixels = self.caption_settings.get('min_image_megapixels')
        max_megapixels = self.caption_settings.get('max_image_megapixels')
        min_pixels = (round(min_megapixels * megapixels_to_pixels)
                      if min_megapixels else DEFAULT_MIN_PIXELS)
        max_pixels = (round(max_megapixels * megapixels_to_pixels)
                      if max_megapixels else DEFAULT_MAX_PIXELS)
        return min_pixels, max_pixels

    def get_size_kwargs(self) -> dict:
        """
        The per-call pixel budget, passed as the image processor's `size`
        dict (`min_pixels`/`max_pixels` processor-call kwargs are
        deprecated in `transformers` and slated for removal). The budget is
        applied per call rather than only at processor construction so
        that changing it does not require reloading the model.
        """
        min_pixels, max_pixels = self.get_pixels()
        return {'size': {'shortest_edge': min_pixels,
                         'longest_edge': max_pixels}}

    def get_processor(self):
        min_pixels, max_pixels = self.get_pixels()
        return AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, min_pixels=min_pixels,
            max_pixels=max_pixels)

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
                                       return_tensors='pt',
                                       **self.get_size_kwargs())
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

    def prepare_reduced_inputs(self, image_prompt: str, image: Image,
                               scale: float):
        """
        One OOM retry at a reduced resolution: an image that doesn't fit
        alongside its KV cache at the configured pixel budget often fits
        at half of it. The floor keeps even the retry above ~64 visual
        tokens so the caption isn't generated from a thumbnail.
        """
        if self.device.type != 'cuda':
            return None
        min_pixels, max_pixels = self.get_pixels()
        reduced_max_pixels = max(64 * PIXELS_PER_VISUAL_TOKEN,
                                 int(max_pixels * scale))
        reduced_min_pixels = min(min_pixels, reduced_max_pixels)
        text = self.get_input_text(image_prompt)
        pil_image = self.load_image(image)
        model_inputs = (self.processor(
                            text=text, images=pil_image, return_tensors='pt',
                            size={'shortest_edge': reduced_min_pixels,
                                  'longest_edge': reduced_max_pixels})
                        .to(self.device, **self.dtype_argument))
        self.input_token_count = model_inputs['input_ids'].shape[-1]
        return model_inputs

    def get_batched_model_inputs(self, batch: list[dict]):
        # Left padding aligns every row's last real token, so generation
        # appends after the same padded length for all rows and the single
        # `input_token_count` slice below is correct for each of them.
        self.processor.tokenizer.padding_side = 'left'
        texts = [self.get_input_text(item['image_prompt']) for item in batch]
        pil_images = [item['pil_image'] for item in batch]
        model_inputs = (self.processor(text=texts, images=pil_images,
                                       padding=True, return_tensors='pt',
                                       **self.get_size_kwargs())
                        .to(self.device, **self.dtype_argument))
        # The padded input length, shared by all rows under left padding.
        self.input_token_count = model_inputs['input_ids'].shape[-1]
        return model_inputs

    def generate_captions_batch(
            self, model_inputs, batch: list[dict]) -> tuple[list[str],
                                                            list[str]]:
        self.tokenizer = self.get_tokenizer()
        parameters = self.get_generation_parameters()
        parameters.update(self.get_cache_kwargs())
        generated_token_ids = self.run_generate(model_inputs, parameters)
        generated_token_ids = (generated_token_ids
                               [:, self.input_token_count:])
        generated_texts = self.processor.batch_decode(
            generated_token_ids, skip_special_tokens=True)
        captions = [self.get_caption_from_generated_text(generated_text,
                                                         item['image_prompt'])
                    for generated_text, item
                    in zip(generated_texts, batch)]
        return captions, captions

    def warmup(self):
        # A short text-only generation warms the decode-path CUDA kernels
        # (the ones per-image runs repeat thousands of times) so the first
        # real image's timing is representative. It cannot run the visual
        # path, but that only costs a few extra seconds on image 1, not the
        # much larger one-time init costs. Best-effort: any failure here
        # would show up identically on the first real image anyway.
        if self.device.type != 'cuda':
            return
        try:
            conversation = [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}
            ]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            model_inputs = (self.processor.tokenizer(text, return_tensors='pt')
                            .to(self.device))
            with torch.inference_mode():
                self.get_generation_model().generate(
                    **model_inputs, max_new_tokens=2, do_sample=False)
        except Exception:
            pass

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
