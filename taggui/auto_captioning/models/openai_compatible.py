import base64
import io
import json
from datetime import datetime
from urllib.request import Request, urlopen

from auto_captioning.auto_captioning_model import (AutoCaptioningModel,
                                                   THINK_BLOCK_PATTERN)
from utils.image import Image

# How long to wait (in seconds) for a single chat completion response before
# giving up. Reasoning models with thinking enabled can take minutes to
# answer, so this is deliberately generous.
REQUEST_TIMEOUT_SECONDS = 900


def post_json(url: str, payload: dict, api_key: str | None,
              timeout: float) -> dict:
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    request = Request(url, data=json.dumps(payload).encode('utf-8'),
                      headers=headers, method='POST')
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def get_json(url: str, api_key: str | None, timeout: float) -> dict:
    headers = {'Accept': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


class OpenAiCompatibleEndpoint(AutoCaptioningModel):
    """
    Caption with a model served by an OpenAI-compatible API server such as
    vLLM or SGLang, instead of loading transformers' implementation in
    this process. This is by far the fastest way to caption large datasets
    with Qwen3-VL-family models: the server runs continuous batching,
    prefix caching, FlashAttention/FlashInfer kernels, and fused MoE
    kernels, all of which are beyond what an in-process transformers
    `generate()` loop can do (especially on Windows, where vLLM/SGLang run
    under WSL2 or Docker).

    The server batches requests internally, so this model "batches" by
    sending several image-captioning requests concurrently (`Batch size`
    controls how many are in flight at once). Each request carries one
    image, encoded as a base64 JPEG data URI.

    No local model is downloaded or loaded; `Batch size`, the prompt, the
    generation parameters, and the caption position all still apply, but
    the device/quantization/SageAttention settings do not.
    """
    is_remote_endpoint = True

    def get_base_url(self) -> str:
        return self.api_base_url.rstrip('/')

    def resolve_api_model_name(self) -> str | None:
        """Resolve the server-side model name, if not set by the user."""
        if self.api_model_name:
            return self.api_model_name
        try:
            response = get_json(f'{self.get_base_url()}/models',
                                self.api_key, REQUEST_TIMEOUT_SECONDS)
        except (OSError, ValueError):
            return None
        try:
            return response['data'][0]['id']
        except (KeyError, IndexError, TypeError):
            return None

    def get_additional_error_message(self) -> str | None:
        if not self.api_base_url:
            return ('No server URL is set for the remote API endpoint. Set '
                    'it in the Auto-Captioner (for example '
                    '`http://127.0.0.1:8000/v1` for a local vLLM server).')
        if not (self.api_base_url.startswith('http://')
                or self.api_base_url.startswith('https://')):
            return (f'The server URL `{self.api_base_url}` is not a valid '
                    f'HTTP(S) URL.')
        if self.beam_count > 1:
            return ('OpenAI-compatible servers do not support beam search. '
                    'Set `Number of beams` to 1 when using a remote API '
                    'endpoint.')
        return None

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        return (f'Captioning with the model served at {self.get_base_url()} '
                f'(start time: '
                f'{self.get_captioning_start_datetime_string(captioning_start_datetime)})')

    def load_processor_and_model(self):
        # No local model to load; just resolve the server-side model name
        # and make sure the server is reachable, so that the run fails fast
        # with a clear message instead of after the first image's upload.
        print(f'Using the model served at {self.get_base_url()}...')
        api_model_name = self.resolve_api_model_name()
        if not api_model_name:
            raise RuntimeError(
                f'Could not reach an OpenAI-compatible server at '
                f'{self.get_base_url()} (the `/models` request failed or '
                f'returned nothing). Start the server (for example `vllm '
                f'serve <model>` or `sglang launch_server --model <model>`) '
                f'and check the URL and API key.')
        self.api_model_name = api_model_name
        print(f'Server model name: {self.api_model_name}')

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> dict:
        # The HTTP request itself happens in `generate_caption`; here only
        # the CPU-side work (image load and base64 encoding) is done, so
        # that it can overlap with in-flight requests.
        pil_image = self.load_image(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        data_uri = ('data:image/jpeg;base64,'
                    + base64.b64encode(buffer.getvalue()).decode('ascii'))
        return {'data_uri': data_uri}

    @staticmethod
    def postprocess_generated_text(generated_text: str) -> str:
        # The server may be serving a hybrid think/no-think model; whether
        # it thinks is under the server's control (there is no portable
        # way to pass `enable_thinking=False` through the OpenAI API), so
        # unconditionally strip any think block from the response.
        return THINK_BLOCK_PATTERN.sub('', generated_text).strip()

    def generate_caption(self, model_inputs: dict,
                         image_prompt: str) -> tuple[str, str]:
        message_content = [
            {'type': 'image_url',
             'image_url': {'url': model_inputs['data_uri']}},
            {'type': 'text', 'text': image_prompt}
        ]
        payload = {
            'model': self.api_model_name,
            'messages': [
                {'role': 'user', 'content': message_content}
            ],
            'max_tokens': self.generation_parameters['max_new_tokens'],
            # OpenAI-style greedy decoding: a temperature of 0 disables
            # sampling; the `do_sample` checkbox toggles between that and
            # the configured temperature.
            'temperature': (self.generation_parameters['temperature']
                            if self.generation_parameters['do_sample'] else 0),
            'top_p': self.generation_parameters['top_p'],
            'top_k': self.generation_parameters['top_k'],
            'repetition_penalty':
                self.generation_parameters['repetition_penalty'],
        }
        if self.generation_parameters['min_new_tokens'] > 1:
            payload['min_tokens'] = (
                self.generation_parameters['min_new_tokens'])
        response = post_json(f'{self.get_base_url()}/chat/completions',
                             payload, self.api_key, REQUEST_TIMEOUT_SECONDS)
        try:
            generated_text = response['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exception:
            raise RuntimeError(
                f'The server returned an unexpected response: {response}.') \
                from exception
        caption = self.get_caption_from_generated_text(generated_text,
                                                       image_prompt)
        return caption, caption
