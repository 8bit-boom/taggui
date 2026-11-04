import gc
import re
import traceback
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image as PilImage
from PIL.ImageOps import exif_transpose
from transformers import (AutoProcessor, BatchFeature, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.utils.import_utils import is_torch_bf16_gpu_available

try:
    # FlashAttention 2 ships no official Windows wheels; on Windows it
    # comes from a community wheel (or Linux pip). Imported defensively so
    # that selecting the `FlashAttention 2` attention implementation
    # without the package fails with a clear message via
    # `get_error_message` instead of a raw `ModuleNotFoundError` at load.
    import flash_attn
    IS_FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    IS_FLASH_ATTENTION_AVAILABLE = False

try:
    # `sageattention` has no official PyPI wheels for Windows, so
    # `requirements.txt` installs a community-built wheel there instead
    # (only available for Python 3.11/3.12); it's not installed at all on
    # macOS. Imported defensively here so that enabling the `Sage
    # attention` option where it's missing fails with a clear message via
    # `get_error_message` below, instead of a raw `ModuleNotFoundError`
    # mid-captioning after the model has already been loaded.
    import sageattention
    IS_SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    IS_SAGE_ATTENTION_AVAILABLE = False

# `apply_sage_attention` below monkey-patches this process-wide, so the
# original implementation is captured here (before any patching can occur)
# to allow reverting it when `Sage attention` is turned back off.
ORIGINAL_SCALED_DOT_PRODUCT_ATTENTION = (
    torch.nn.functional.scaled_dot_product_attention)

try:
    # `AutoModelForVision2Seq` was renamed to `AutoModelForImageTextToText` in
    # transformers 5.0.
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq

from utils.enums import CaptionDevice
from utils.image import Image

if TYPE_CHECKING:
    # Only needed for the `captioning_thread_` type hint below. Importing
    # this at runtime would create a circular import with
    # `captioning_thread.py`, which imports `AutoCaptioningModel` from this
    # module.
    import auto_captioning.captioning_thread as captioning_thread


def replace_template_variable(match: re.Match, image: Image) -> str:
    template_variable = match.group(0)[1:-1].lower()
    if template_variable == 'tags':
        return ', '.join(image.tags)
    if template_variable == 'name':
        return image.path.stem
    if template_variable in ('directory', 'folder'):
        return image.path.parent.name


def replace_template_variables(text: str, image: Image) -> str:
    # Replace template variables inside curly braces that are not escaped.
    text = re.sub(r'(?<!\\){[^{}]+(?<!\\)}',
                  lambda match: replace_template_variable(match, image), text)
    # Unescape escaped curly braces.
    text = re.sub(r'\\([{}])', r'\1', text)
    return text


# Reasoning models (the Qwen3-VL family in particular) may emit a
# `<think>...</think>` block before the actual caption. The opening
# `<think>` is often forced by the chat template as part of the generation
# prompt rather than actually generated, so the decoded output often starts
# mid-reasoning with no opening tag at all - only the closing `</think>`.
# This pattern matches both cases. Defined here (not in the Qwen3-VL
# module) so that the remote-endpoint model can strip think blocks too.
THINK_BLOCK_PATTERN = re.compile(r'(?:<think>)?.*?</think>\s*', re.DOTALL)


class CancelCaptioningWhenRequested(StoppingCriteria):
    """
    Stops `generate()` as soon as the captioning thread's `is_canceled`
    flag is set, so that canceling doesn't have to wait for the current
    image's full token budget (which can be minutes for a reasoning
    model). The partial generation is discarded by the captioning thread.
    """

    def __init__(self, captioning_thread):
        self.captioning_thread = captioning_thread

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.captioning_thread.is_canceled

# Caption-settings value -> `from_pretrained(attn_implementation=...)`
# value. `Sage attention` is not an `attn_implementation` because it works
# by monkey-patching `torch.nn.functional.scaled_dot_product_attention`
# (see `apply_sage_attention` below), which no `attn_implementation` value
# can express.
ATTENTION_IMPLEMENTATIONS = {
    'Default': None,
    'SDPA': 'sdpa',
    'FlashAttention 2': 'flash_attention_2'
}


class AutoCaptioningModel:
    dtype = torch.float16
    # When loading a model, if the `use_safetensors` argument is not set and
    # both a safetensors and a non-safetensors version of the model are
    # available, both versions get downloaded. This should be set to `None` for
    # models that do not have a safetensors version.
    use_safetensors = True
    model_load_context_manager = nullcontext()
    transformers_model_class = AutoModelForVision2Seq
    image_mode = 'RGB'
    # Whether `generate_captions_batch` is implemented for this model, i.e.
    # whether several images can be captioned in a single `generate()` call.
    # Batching is the single biggest in-process speedup (decoding is
    # memory-bandwidth-bound, so many images cost barely more than one),
    # but it requires model-specific input padding, so it is opt-in per
    # model class. The remote-endpoint model "batches" differently, by
    # sending concurrent requests (see `is_remote_endpoint`).
    supports_batching = False
    is_remote_endpoint = False

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        self.thread = captioning_thread_
        self.thread_parent = captioning_thread_.parent()
        self.caption_settings = caption_settings
        self.model_id = caption_settings['model_id']
        self.prompt = caption_settings['prompt']
        self.caption_start = caption_settings['caption_start']
        self.device_setting: CaptionDevice = caption_settings['device']
        self.device: torch.device = self.get_device()
        if self.dtype == torch.bfloat16:
            if self.device.type != 'cuda' or not is_torch_bf16_gpu_available():
                self.dtype = torch.float16
        self.dtype_argument = ({'dtype': self.dtype}
                               if self.device.type == 'cuda' else {})
        self.load_in_4_bit = caption_settings['load_in_4_bit']
        self.load_in_8_bit = caption_settings['load_in_8_bit']
        self.cpu_offload = caption_settings['cpu_offload']
        self.sage_attention = caption_settings['sage_attention']
        # Only meaningful for models whose `format_prompt()` reads it (the
        # Qwen3-VL family, as of writing); harmless to always set it here so
        # every model follows the same `caption_settings[key]` contract.
        self.disable_thinking = caption_settings['disable_thinking']
        self.wd_tagger_load_in_8_bit = caption_settings[
            'wd_tagger_load_in_8_bit']
        self.bad_words_string = caption_settings['bad_words']
        self.forced_words_string = caption_settings['forced_words']
        self.remove_tag_separators = caption_settings['remove_tag_separators']
        self.generation_parameters = caption_settings['generation_parameters']
        self.beam_count = self.generation_parameters['num_beams']
        # `get()` with defaults keeps captioning profiles saved by older
        # versions of the app loadable.
        self.batch_size = max(1, int(caption_settings.get('batch_size', 1)
                                     or 1))
        self.attention_implementation = caption_settings.get(
            'attention_implementation', 'Default')
        self.torch_compile = bool(caption_settings.get('torch_compile', False))
        self.quantize_kv_cache = bool(
            caption_settings.get('quantize_kv_cache', False))
        self.api_base_url = caption_settings.get('api_base_url', '').strip()
        self.api_key = caption_settings.get('api_key', '').strip()
        self.api_model_name = caption_settings.get('api_model_name',
                                                   '').strip()
        # The tokenizer is not available until the processor is loaded, so
        # the `bad_words`/`forced_words` tokenization is deferred; these
        # are then computed once per model instance (the strings cannot
        # change mid-run), instead of being re-tokenized for every image.
        self._bad_words_ids = None
        self._forced_words_ids = None
        # Set by `load_processor_and_model` when the checkpoint ships with
        # its own quantization (GPTQ/AWQ/bitsandbytes/...), which is used
        # instead of this app's bitsandbytes options.
        self.checkpoint_quantization: str | None = None
        self.processor = None
        self.model = None
        self.tokenizer = None

    def get_device(self) -> torch.device:
        # `Multiple GPUs (parallel)` uses the configured GPU index as its
        # primary device; the other devices are created per worker by the
        # captioning thread.
        if (self.device_setting in (CaptionDevice.GPU,
                                    CaptionDevice.GPUS_PARALLEL)
                and torch.cuda.is_available()):
            gpu_index = self.caption_settings['gpu_index']
            device = torch.device(f'cuda:{gpu_index}')
        else:
            device = torch.device('cpu')
        return device

    def get_additional_error_message(self) -> str | None:
        return None

    def get_error_message(self) -> str | None:
        # The remote-endpoint model has no local load, so the device,
        # quantization, and attention checks below don't apply to it; it
        # validates its own settings in `get_additional_error_message`.
        if self.is_remote_endpoint:
            return self.get_additional_error_message()
        if self.forced_words_string.strip() and self.beam_count < 2:
            return ('`Number of beams` must be greater than 1 when `Include '
                    'in caption` is not empty.')
        if (self.sage_attention and self.device.type == 'cuda'
                and not IS_SAGE_ATTENTION_AVAILABLE):
            return ('The `sageattention` package is not installed. It is '
                    'installed automatically on Linux and on Windows with '
                    'Python 3.11 or 3.12; on macOS, or with another Python '
                    'version on Windows, install it manually to use `Sage '
                    'attention`, or disable that option.')
        if (self.attention_implementation == 'FlashAttention 2'
                and self.device.type == 'cuda'
                and not IS_FLASH_ATTENTION_AVAILABLE):
            return ('The `flash_attn` package is not installed, so the '
                    '`FlashAttention 2` attention implementation cannot be '
                    'used. Install it (`pip install flash-attn --no-build-'
                    'isolation`, or a community wheel on Windows), or set '
                    '`Attention implementation` back to `Default`.')
        if self.torch_compile:
            if self.device.type != 'cuda':
                return ('`Compile model` requires a GPU.')
            if self.load_in_4_bit or self.load_in_8_bit:
                return ('`Compile model` is not compatible with '
                        'bitsandbytes 4-bit/8-bit quantization. Use an '
                        'unquantized or pre-quantized (GPTQ/AWQ) '
                        'checkpoint instead, or disable `Compile model`.')
            if self.sage_attention:
                return ('`Compile model` is not compatible with `Sage '
                        'attention` (SageAttention monkey-patches the '
                        'attention function that torch.compile traces). '
                        'Enable only one of the two.')
        if self.quantize_kv_cache and self.device.type == 'cuda':
            try:
                from transformers.cache_utils import Fp8QuantizedLayer
            except ImportError:
                return ('`Quantize KV cache` needs a newer version of the '
                        '`transformers` package (one that ships the `fp8` '
                        'cache backend). Upgrade it with `pip install -U '
                        'transformers`, or disable `Quantize KV cache`.')
        elif self.quantize_kv_cache:
            return ('`Quantize KV cache` requires a GPU.')
        return self.get_additional_error_message()

    def get_processor(self):
        return AutoProcessor.from_pretrained(self.model_id,
                                             trust_remote_code=True)

    @staticmethod
    def get_quantization_skip_modules() -> list[str] | None:
        """
        Module names to exclude from 4-bit/8-bit quantization (matched as a
        substring of each module's full dotted name). `None` quantizes
        every `nn.Linear` layer, which is fine for a text-only model, but a
        multimodal model's vision tower and output head are usually kept
        in full precision - quantizing them doesn't necessarily crash, but
        is known to noticeably hurt caption quality for not much VRAM
        savings, since most of a vision-language model's weights are in
        the language model to begin with.
        """
        return None

    def get_checkpoint_quantization(self) -> str | None:
        """
        Return the name of the quantization method that the checkpoint
        itself ships with (GPTQ, AWQ, bitsandbytes, ...), or `None` if the
        checkpoint is unquantized. Pre-quantized checkpoints are applied
        automatically by `transformers` when loading them, and they use
        faster kernels (e.g. Marlin for GPTQ/AWQ int4) than this app's
        bitsandbytes options, so the `Load in 4-bit`/`Load in 8-bit`
        checkboxes are ignored for them instead of stacking a second
        quantization on top of the first.
        """
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_id,
                                                trust_remote_code=True)
        except Exception:
            return None
        quantization_config = getattr(config, 'quantization_config', None)
        if not quantization_config:
            return None
        if isinstance(quantization_config, dict):
            return quantization_config.get('quant_method', 'pre-quantized')
        return getattr(quantization_config, 'quant_method',
                       type(quantization_config).__name__)

    def get_model_load_arguments(self) -> dict:
        device_map = 'auto' if self.cpu_offload else self.device
        arguments = {'device_map': device_map, 'trust_remote_code': True,
                     'use_safetensors': self.use_safetensors}
        skip_modules = self.get_quantization_skip_modules()
        if self.load_in_4_bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_storage=self.dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip_modules
            )
            arguments['quantization_config'] = quantization_config
        elif self.load_in_8_bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=skip_modules)
            arguments['quantization_config'] = quantization_config
        if self.device.type == 'cuda':
            arguments['torch_dtype'] = self.dtype
        attention_implementation = ATTENTION_IMPLEMENTATIONS.get(
            self.attention_implementation)
        # `flash_attention_2` is CUDA-only; `sdpa` is a no-op on CPU
        # (SDPA is already the CPU path).
        if (attention_implementation and self.device.type == 'cuda'):
            arguments['attn_implementation'] = attention_implementation
        return arguments

    def apply_sage_attention(self):
        # This patch is process-wide (not scoped to this model instance), so
        # it must be explicitly reverted when SageAttention isn't requested,
        # not just left unset - otherwise a model loaded earlier with it
        # enabled would leave every later model (including ones that never
        # asked for it) silently running through `sageattn`.
        if self.sage_attention and self.device.type == 'cuda':
            torch.nn.functional.scaled_dot_product_attention = (
                sageattention.sageattn)
        else:
            torch.nn.functional.scaled_dot_product_attention = (
                ORIGINAL_SCALED_DOT_PRODUCT_ATTENTION)

    def load_model(self, model_load_arguments: dict):
        with self.model_load_context_manager:
            model = self.transformers_model_class.from_pretrained(
                self.model_id, **model_load_arguments)
        model.eval()
        self.apply_sage_attention()
        return model

    def patch_source_code(self) -> bool:
        # Return `True` if the source code was patched.
        return False

    def get_model(self):
        model_load_arguments = self.get_model_load_arguments()
        model = self.load_model(model_load_arguments)
        if self.patch_source_code():
            print('Patched the model source code. Reloading the model...')
            model = self.load_model(model_load_arguments)
        return model

    def load_processor_and_model(self):
        models_directory_path = self.thread.models_directory_path
        if models_directory_path:
            config_path = models_directory_path / self.model_id / 'config.json'
            tags_path = (models_directory_path / self.model_id
                         / 'selected_tags.csv')
            if config_path.is_file() or tags_path.is_file():
                self.model_id = str(models_directory_path / self.model_id)
        # If the processor and model were previously loaded, use them.
        processor = self.thread_parent.processor
        model = self.thread_parent.model
        # Only GPUs support 4-bit and 8-bit quantization.
        self.load_in_4_bit = self.load_in_4_bit and self.device.type == 'cuda'
        self.load_in_8_bit = (self.load_in_8_bit and not self.load_in_4_bit
                              and self.device.type == 'cuda')
        if (model and self.thread_parent.model_id == self.model_id
                and (self.thread_parent.model_device_type
                     == self.device.type)
                and (self.thread_parent.is_model_loaded_in_4_bit
                     == self.load_in_4_bit)
                and (self.thread_parent.is_model_loaded_in_8_bit
                     == self.load_in_8_bit)
                and (self.thread_parent.is_model_cpu_offloaded
                     == self.cpu_offload)
                and (self.thread_parent.is_model_using_sage_attention
                     == self.sage_attention)
                and (self.thread_parent.is_model_torch_compiled
                     == self.torch_compile)
                and (self.thread_parent.is_model_attention_implementation
                     == self.attention_implementation)
                and (self.thread_parent.is_wd_tagger_loaded_in_8_bit
                     == self.wd_tagger_load_in_8_bit)):
            self.processor = processor
            self.model = model
            return
        # Load the new processor and model.
        if model:
            # Garbage collect the previous processor and model to free up
            # memory.
            self.thread_parent.processor = None
            self.thread_parent.model = None
            del processor
            del model
            gc.collect()
        self.thread.clear_console_text_edit_requested.emit()
        print(f'Loading {self.model_id}...')
        # Check whether the checkpoint is pre-quantized before the bnb
        # options are folded into the load arguments below, so a
        # pre-quantized checkpoint is never double-quantized.
        self.checkpoint_quantization = self.get_checkpoint_quantization()
        if self.checkpoint_quantization:
            if self.load_in_4_bit or self.load_in_8_bit:
                print(f'This checkpoint is already quantized '
                      f'({self.checkpoint_quantization}); its built-in '
                      f'quantization kernels will be used and `Load in '
                      f'4-bit`/`Load in 8-bit` is ignored.')
            else:
                print(f'This checkpoint is pre-quantized '
                      f'({self.checkpoint_quantization}); its built-in '
                      f'quantization kernels will be used.')
            self.load_in_4_bit = False
            self.load_in_8_bit = False
        self.processor = self.get_processor()
        self.thread_parent.processor = self.processor
        self.model = self.get_model()
        if self.torch_compile and self.device.type == 'cuda':
            print('Compiling the model with torch.compile (the first run '
                  'will be slower while it compiles)...')
            # `dynamic=True` avoids recompiling for every distinct image
            # resolution and sequence length.
            self.model = torch.compile(self.model, dynamic=True)
        self.thread_parent.model = self.model
        self.warmup()
        self.thread_parent.model_id = self.model_id
        self.thread_parent.model_device_type = self.device.type
        self.thread_parent.is_model_loaded_in_4_bit = self.load_in_4_bit
        self.thread_parent.is_model_loaded_in_8_bit = self.load_in_8_bit
        self.thread_parent.is_model_cpu_offloaded = self.cpu_offload
        self.thread_parent.is_model_using_sage_attention = (
            self.sage_attention)
        self.thread_parent.is_model_torch_compiled = self.torch_compile
        self.thread_parent.is_model_attention_implementation = (
            self.attention_implementation)
        self.thread_parent.is_wd_tagger_loaded_in_8_bit = (
            self.wd_tagger_load_in_8_bit)

    def warmup(self):
        """
        Called once after a (re)load so that the first real image does not
        pay for CUDA kernel autotuning and other one-time initialization
        (which would otherwise skew its per-image timing and any average
        derived from it). The base implementation does nothing.
        """
        return

    def monkey_patch_after_loading(self):
        return

    @staticmethod
    def get_captioning_start_datetime_string(
            captioning_start_datetime: datetime) -> str:
        return captioning_start_datetime.strftime('%Y-%m-%d %H:%M:%S')

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return (f'Captioning... (device: {self.device}, start time: '
                    f'{captioning_start_datetime_string})')
        return f'Captioning... (device: {self.device})'

    @staticmethod
    def get_default_prompt() -> str:
        return ''

    @staticmethod
    def format_prompt(prompt: str) -> str:
        return prompt

    def get_image_prompt(self, image: Image) -> str | None:
        if self.prompt:
            image_prompt = replace_template_variables(self.prompt, image)
        else:
            self.prompt = self.get_default_prompt()
            image_prompt = self.prompt
        image_prompt = self.format_prompt(image_prompt)
        return image_prompt

    def get_input_text(self, image_prompt: str) -> str:
        if image_prompt and self.caption_start:
            text = f'{image_prompt} {self.caption_start}'
        else:
            text = image_prompt or self.caption_start
        return text

    def load_image(self, image: Image) -> PilImage:
        pil_image = PilImage.open(image.path)
        # Rotate the image according to the orientation tag.
        pil_image = exif_transpose(pil_image)
        pil_image = pil_image.convert(self.image_mode)
        return pil_image

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> BatchFeature | dict | np.ndarray:
        text = self.get_input_text(image_prompt)
        pil_image = self.load_image(image)
        model_inputs = (self.processor(text=text, images=pil_image,
                                       return_tensors='pt')
                        .to(self.device, **self.dtype_argument))
        return model_inputs

    def get_generation_model(self):
        return self.model

    def get_tokenizer(self):
        return self.processor.tokenizer

    def get_bad_words_ids(self) -> list[list[int]] | None:
        if not self.bad_words_string.strip():
            return None
        if self._bad_words_ids is None:
            words = re.split(r'(?<!\\),', self.bad_words_string)
            words = [word.strip() for word in words if word.strip()]
            if not words:
                return None
            words = [word.replace(r'\,', ',') for word in words]
            # Also discourage generating the versions of the words with
            # spaces before them.
            words += [' ' + word for word in words]
            self._bad_words_ids = self.tokenizer(
                words, add_special_tokens=False).input_ids
        return self._bad_words_ids

    def get_forced_words_ids(self) -> list[list[list[int]]] | None:
        if not self.forced_words_string.strip():
            return None
        if self._forced_words_ids is None:
            word_groups = re.split(r'(?<!\\),', self.forced_words_string)
            forced_words_ids = []
            for word_group in word_groups:
                word_group = word_group.strip().replace(r'\,', ',')
                words = re.split(r'(?<!\\)\|', word_group)
                words = [word.strip() for word in words if word.strip()]
                if not words:
                    continue
                words = [word.replace(r'\|', '|') for word in words]
                words_ids = self.tokenizer(words,
                                           add_special_tokens=False).input_ids
                forced_words_ids.append(words_ids)
            self._forced_words_ids = (forced_words_ids or None)
        return self._forced_words_ids

    @staticmethod
    def get_additional_generation_parameters() -> dict:
        return {}

    @staticmethod
    def postprocess_image_prompt(image_prompt: str) -> str:
        return image_prompt

    @staticmethod
    def postprocess_generated_text(generated_text: str) -> str:
        return generated_text

    def get_caption_from_generated_tokens(
            self, generated_token_ids: torch.Tensor, image_prompt: str) -> str:
        generated_text = self.processor.batch_decode(
            generated_token_ids, skip_special_tokens=True)[0]
        return self.get_caption_from_generated_text(generated_text,
                                                    image_prompt)

    def get_caption_from_generated_text(self, generated_text: str,
                                        image_prompt: str) -> str:
        image_prompt = self.postprocess_image_prompt(image_prompt)
        generated_text = self.postprocess_generated_text(generated_text)
        if image_prompt.strip() and generated_text.startswith(image_prompt):
            caption = generated_text[len(image_prompt):]
        elif (self.caption_start.strip()
              and generated_text.startswith(self.caption_start)):
            caption = generated_text
        else:
            caption = f'{self.caption_start.strip()} {generated_text.strip()}'
        caption = caption.strip()
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')
        return caption

    @staticmethod
    def is_exception_from_sage_attention(exception: BaseException) -> bool:
        return any('sageattention' in (frame.filename or '').lower()
                  for frame in traceback.extract_tb(exception.__traceback__))

    def get_generation_parameters(self) -> dict:
        """
        Assemble the `generate()` keyword arguments from the user's
        settings, computing the (memoized) `bad_words`/`forced_words` token
        IDs once instead of re-tokenizing them for every image.
        """
        parameters = {**self.generation_parameters,
                      **self.get_additional_generation_parameters(),
                      'bad_words_ids': self.get_bad_words_ids(),
                      'force_words_ids': self.get_forced_words_ids()}
        return parameters

    def get_cache_kwargs(self) -> dict:
        """
        Extra `generate()` keyword arguments for the key-value cache. The
        `fp8` quantized cache stores the cache in 8-bit float (via
        `torch.float8_e4m3fn`, no extra package needed), roughly halving
        cache memory - which is what buys the headroom for larger batch
        sizes - at a small per-token dequantization cost.
        """
        if self.quantize_kv_cache and self.device.type == 'cuda':
            return {'cache_implementation': 'quantized',
                    'cache_config': {'backend': 'fp8'}}
        return {}

    def run_generate(self, model_inputs: BatchFeature | dict | np.ndarray,
                     parameters: dict) -> torch.Tensor:
        # Stop mid-generation as soon as the user cancels, instead of
        # finishing the current image's whole token budget first. The
        # partial generation is discarded by the captioning thread.
        parameters.setdefault('stopping_criteria', StoppingCriteriaList(
            [CancelCaptioningWhenRequested(self.thread)]))
        with torch.inference_mode():
            try:
                return self.get_generation_model().generate(**model_inputs,
                                                            **parameters)
            except Exception as exception:
                # SageAttention's CUDA kernels only support certain
                # attention head sizes; a model whose architecture doesn't
                # match one raises an opaque error deep inside
                # `sageattention` itself (e.g. `Unsupported head_dim: 256`)
                # instead of a clear, actionable one.
                if (self.sage_attention
                        and self.is_exception_from_sage_attention(exception)):
                    raise RuntimeError(
                        f'SageAttention failed to run with this model '
                        f'({type(exception).__name__}: {exception}). Its '
                        f'CUDA kernels only support certain attention head '
                        f'sizes, and this model isn\'t compatible. Disable '
                        f'`Sage attention` in the Auto-Captioner settings '
                        f'to use this model.') from exception
                raise

    def generate_caption(self, model_inputs: BatchFeature | dict | np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        self.tokenizer = self.get_tokenizer()
        parameters = self.get_generation_parameters()
        parameters.update(self.get_cache_kwargs())
        generated_token_ids = self.run_generate(model_inputs, parameters)
        caption = self.get_caption_from_generated_tokens(generated_token_ids,
                                                         image_prompt)
        console_output_caption = caption
        return caption, console_output_caption

    def get_batched_model_inputs(
            self, batch: list[dict]) -> BatchFeature | dict | np.ndarray:
        """
        Given a list of prepared images (each a dict with at least
        `image_prompt` and `pil_image`), build a single batched input for
        them. Only models with `supports_batching = True` need to implement
        this; the captioning thread never calls it otherwise.
        """
        raise NotImplementedError

    def generate_captions_batch(
            self, model_inputs: BatchFeature | dict | np.ndarray,
            batch: list[dict]) -> tuple[list[str], list[str]]:
        """
        Caption a batch of images (prepared by `get_batched_model_inputs`)
        in a single `generate()` call. Returns a caption and a console-
        output caption per image, in the same order as `batch`.
        """
        raise NotImplementedError

    def prepare_reduced_inputs(
            self, image_prompt: str,
            image: Image, scale: float) -> BatchFeature | dict | None:
        """
        Re-prepare one image's inputs with a smaller pixel budget (the
        image's dimensions multiplied by `scale`), used to retry an image
        once after a CUDA out-of-memory error. Returns `None` for models
        that don't support this (the image is then reported as failed).
        """
        return None
