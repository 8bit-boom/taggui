# Fork notes

This is a fork of [jhc13/taggui](https://github.com/jhc13/taggui), a
PySide6 desktop app for tagging and auto-captioning image datasets. This
file tracks what's different here and why, for anyone (human or AI) picking
this codebase back up later.

The fork branched off upstream at `cb8cca7` ("Merge pull request #378 from
jhc13/update-packages") and has since added new auto-captioning models,
quantization/performance options, dataset-management tooling, and a batch
of real-world bug fixes found by actually running the app against current
model checkpoints rather than just reading documentation.

## Why this fork exists

The driving use case is auto-captioning datasets with newer, bigger
vision-language models (Qwen3-VL, Qwen3.5 and its later point releases) on
a local Windows GPU box, plus day-to-day dataset-curation features
(filtering, statistics, tag presets, export) that weren't in upstream.
Everything here was built and debugged against a real install, with
tracebacks from an actual Windows + NVIDIA GPU setup driving most of the
bug fixes.

## Feature additions

### Auto-captioning models and performance
- **Qwen3-VL** and **Qwen3.5** support, plus automatic recognition of
  later point releases in the same architecture family (Qwen3.6, Qwen3.8,
  and future `Qwen3.x` releases) without needing a code change per
  release - see `taggui/auto_captioning/models_list.py`.
- **Qwen3.5 MoE** ("-A3B" checkpoints, e.g. `Qwen3.6-35B-A3B`) as a
  separate model class (`Qwen3_5Moe`), since MoE checkpoints use a
  different `transformers` model class than the dense ones.
- **4-bit/8-bit quantization** that excludes the vision tower and output
  head on Qwen3-VL-family models (`get_quantization_skip_modules()`),
  matching the `llm_int8_skip_modules` convention used by published
  4-bit/8-bit Qwen3-VL checkpoints - quantizing those layers measurably
  hurts caption quality for little VRAM savings.
- **SageAttention** support with automatic installation on Windows
  (Python 3.11/3.12) via a community-built wheel referenced directly in
  `requirements.txt`, since `sageattention` has no official PyPI wheels
  for Windows. Enabling it without the package installed (or with a
  model whose attention head size SageAttention's CUDA kernels don't
  support) now raises a clear, actionable message instead of a crash or
  an unhandled internal stack trace.
- **"Disable thinking"** checkbox for Qwen3-VL-family models, wired to
  `enable_thinking=False` in the chat template (the documented way to
  turn off a hybrid think/no-think model's reasoning step).
- **Prompt template picker**: pick a target image-generation model
  (SDXL, Flux.1, Flux.2, Z-Image, Krea 2 Turbo, Anima) and a purpose
  (General/Character/Style/Landscape/Weapons) to insert a researched
  starting-point prompt, optionally combined with a captioning-model
  suffix (Qwen3-VL/Qwen3.5/JoyCaption) tuned to that model's own response
  quirks. See `taggui/auto_captioning/prompt_templates.py` for the full
  rationale behind each template's wording (why Style-purpose prompts
  omit style words, why SDXL stays short, etc.).
- **Generation parameter presets** (`taggui/auto_captioning/generation_presets.py`):
  one-click preset buttons for the Minimum/Maximum tokens, sampling,
  temperature, top-k/p, etc. widgets, tuned per captioning model - a
  reasoning model needs a much larger token budget than a direct
  captioner, since its `<think>` block consumes most of it before the
  actual answer starts.
- **WD Tagger 8-bit quantization**, restricted to `MatMul` nodes only
  (quantizing the `ConvInteger` nodes in WD Tagger's ConvNeXt/EVA02
  backbones crashes some ONNX Runtime builds at inference time).

### Image format and conversion
- **AVIF image support** (viewing, thumbnails, dimension reading) via
  `pillow-avif-plugin`.
- **Convert Images** dialog for batch re-encoding a dataset to a
  different format.
- **Danbooru tag autocomplete**, loaded from a user-supplied CSV.

### Dataset management
- **Extended image-list filters**: width/height/area/megapixels/aspect
  ratio/file size/extension, plus saved filters with a quick-access menu.
- **Dataset Statistics dock**: tagged/untagged counts, tag frequency
  stats, token-count distribution, aspect-ratio buckets, duplicate
  captions, format/size breakdown, with "Show" buttons that jump to a
  matching image-list filter.
- **Tag Presets dock**: named groups of tags applied/removed in one
  click or via `Alt+1`-`Alt+9`.
- **Captioning Profiles**: save/load/delete a full snapshot of the
  Auto-Captioner settings form.
- **Export Dataset** dialog: CSV/JSON/JSONL (`metadata.jsonl`)/tag
  frequency CSV/file list, respecting the current filter scope.
- **pytest suite** (`tests/`) covering the pure/headless-safe logic most
  likely to silently corrupt a dataset if it regresses - batch tag
  operations, filter parsing, model dispatch, dataset statistics.

### UI fixes
- Fixed a persistent Auto-Captioner dock content-clipping bug, root-caused
  to `QComboBox`'s default `AdjustToContentsOnFirstShow` policy
  permanently widening the form based on a long local model path.
- Added `View > Reset Window Layout` to recover from a stuck/corrupted
  dock layout without reinstalling.

### Windows installation
- `install.bat`/`taggui.bat`/`update.bat` convenience scripts.
- `requirements.txt` now pulls platform-specific wheels for `torch`,
  `flash-attn`, and `sageattention` on Windows directly from their
  release URLs (mirroring how the upstream project already did this for
  `torch`), so a fresh `install.bat` run needs no manual wheel-hunting.

## Notable bugs found and fixed along the way

These are worth knowing about if you're extending the Qwen3-VL-family
model code, since the underlying `transformers` library's class layout
for this model family has shifted more than once during this fork's
lifetime:

- **Qwen3.5's multimodal class changed.** An early `transformers` release
  had `Qwen3_5ForCausalLM` double as the multimodal entry point, needing
  a manual image-embedding merge before `generate()`. A later release
  turned `Qwen3_5ForCausalLM` into a text-only class (no visual encoder
  at all) and moved multimodal support to `Qwen3_5ForConditionalGeneration`,
  which accepts `pixel_values` natively like `Qwen3VLForConditionalGeneration`
  already did. Selecting the wrong one raises
  `AttributeError: 'Qwen3_5ForCausalLM' object has no attribute 'visual'`.
- **Prompt text leaking into captions, twice, for two different reasons.**
  First, a hardcoded list of special-token strings used to strip the
  echoed prompt from Qwen3-VL's output didn't match Qwen3.5's actual
  special tokens (a completely different ID range/vocabulary, not just
  different IDs for the same tokens). Fixed by slicing the generated
  token IDs by input length instead of string-matching - this doesn't
  depend on knowing any tokenizer's special tokens at all. Second, once
  that slicing was in place, the `<think>...</think>` block stripping
  broke in the opposite direction: the chat template forces the opening
  `<think>` tag into the *prompt* (now sliced away), leaving only a bare
  closing `</think>` in the decoded output with no opening tag for the
  strip pattern to match against.
- **SageAttention monkey-patches a process-global PyTorch function**
  (`torch.nn.functional.scaled_dot_product_attention`) and never
  reverted it when turned back off, so every model loaded afterward -
  including ones that never asked for SageAttention - silently kept
  running through it until the app restarted.
- **Export Dataset's "Tag frequency CSV" ignored the Scope setting**,
  always reading from the dataset-wide tag counter instead of the
  actually-selected/filtered image scope.

## Known limitations / things to verify on real hardware

Most of this fork's model-loading and generation-path code was written
and debugged without a GPU available in the development environment,
using `transformers`'/`torch`'s real Python source (installed from PyPI)
to verify class structures, signatures, and config attributes precisely,
plus synthetic unit tests simulating the exact failure traces reported
from a real Windows + NVIDIA setup - but never a full end-to-end model
download and generation run. If you hit something that doesn't match
what's documented here, the `transformers` library's class layout for a
given model may have shifted again since this was written; check the
installed version's actual source for the model in question before
assuming the fork's code is wrong.

- SageAttention's supported attention head sizes aren't consistently
  documented across versions/GPU architectures, so the fix here is
  reactive (catching the crash and giving a clear message) rather than
  predicting compatibility ahead of time.
- The prompt templates in `prompt_templates.py` are researched
  starting points (drawn from official docs and community LoRA-training
  guides), not verified optimal settings for any specific checkpoint.
