"""
In-app local model inference via embedded llama.cpp (llama-cpp-python).

Runs GGUF models directly inside the app - no external server, URL, or API
key. Used by the AI comparison judge so a powerful local LLM (e.g. a
Qwen3.8-27B or Gemma 4 GGUF) can rank captions with nothing but this app.

The prebuilt CUDA wheel of llama-cpp-python needs the CUDA 12 runtime DLLs;
on Windows we add torch's bundled CUDA 12 DLL directory to the search path
before importing llama_cpp so the wheel loads without a separate CUDA install.
"""

import os
import sys
from pathlib import Path

from auto_captioning.models.openai_compatible import THINK_BLOCK_PATTERN

# Loaded Llama instances are expensive (a 27B GGUF is many GB); cache them by
# (path, n_gpu_layers, n_ctx) so repeated judge runs reuse the loaded model.
_LOADED_MODELS: dict[tuple, object] = {}


def ensure_cuda_dll_path() -> None:
    """
    On Windows, make torch's bundled CUDA 12 runtime DLLs discoverable so the
    prebuilt CUDA llama-cpp-python wheel loads. No-op elsewhere / if torch is
    absent. Safe to call repeatedly.
    """
    if sys.platform != 'win32':
        return
    try:
        import torch
        lib_dir = Path(torch.__file__).parent / 'lib'
        if lib_dir.is_dir():
            os.add_dll_directory(str(lib_dir))
    except (ImportError, OSError, AttributeError):
        pass


def is_available() -> bool:
    """Whether llama-cpp-python can be imported (after fixing the DLL path)."""
    ensure_cuda_dll_path()
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False


def discover_gguf_models(extra_directories: list[Path] | None = None) -> list[Path]:
    """
    Find standalone GGUF model files (excluding mmproj vision projectors) in
    the Hugging Face cache, the app's models directory, and any extra dirs
    (e.g. Unsloth outputs). Sorted, de-duplicated.
    """
    roots: list[Path] = [Path.home() / '.cache' / 'huggingface' / 'hub']
    if extra_directories:
        roots.extend(extra_directories)
    found: dict[Path, None] = {}
    for root in roots:
        if not root or not root.is_dir():
            continue
        for path in root.rglob('*.gguf'):
            if 'mmproj' in path.name.lower():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            found[resolved] = None
    return sorted(found)


def load_local_model(model_path: str, n_gpu_layers: int = -1,
                     n_ctx: int = 8192):
    """
    Load (or reuse) a GGUF model in-process. Raises RuntimeError with a clear
    message if llama-cpp-python is missing or the model fails to load.
    """
    ensure_cuda_dll_path()
    try:
        from llama_cpp import Llama
    except Exception as exception:
        raise RuntimeError(
            'llama-cpp-python is not installed, so in-app local models are '
            'unavailable. Install it with: pip install llama-cpp-python '
            '--index-url https://abetlen.github.io/llama-cpp-python/whl/cu124'
        ) from exception
    key = (str(model_path), int(n_gpu_layers), int(n_ctx))
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]
    try:
        model = Llama(model_path=str(model_path), n_gpu_layers=n_gpu_layers,
                      n_ctx=n_ctx, verbose=False)
    except Exception as exception:
        raise RuntimeError(
            f'Failed to load the GGUF model {model_path}: {exception}'
        ) from exception
    _LOADED_MODELS[key] = model
    return model


def release_local_models() -> None:
    """Drop cached in-app models to free their memory/VRAM."""
    _LOADED_MODELS.clear()


def judge_captions_local(model_path: str, judge_prompt: str,
                         n_gpu_layers: int = -1, n_ctx: int = 8192) -> str:
    """
    Run the judging prompt through an in-app GGUF model (no server) and return
    the raw response text. Greedy decoding for a deterministic, parseable
    verdict; think blocks are left for the parser to strip.
    """
    if not model_path or not Path(model_path).is_file():
        raise RuntimeError(
            'No local GGUF model selected. Choose a .gguf file (the dialog '
            'lists models found in your Hugging Face cache), or install one.')
    model = load_local_model(model_path, n_gpu_layers, n_ctx)
    response = model.create_chat_completion(
        messages=[{'role': 'user', 'content': judge_prompt}],
        max_tokens=2000, temperature=0)
    try:
        return response['choices'][0]['message']['content'] or ''
    except (KeyError, IndexError, TypeError) as exception:
        raise RuntimeError(
            f'The local model returned an unexpected response: {response}'
        ) from exception
