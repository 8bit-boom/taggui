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
- **Per-model settings**: switching the `Model` dropdown now saves the
  outgoing model's full settings snapshot (prompt, generation parameters,
  quantization, everything `get_caption_settings()` covers) and restores
  the newly-selected model's own saved settings, if any - so each model
  keeps its own configuration instead of sharing one global set
  (`taggui/auto_captioning/model_settings.py`,
  `CaptionSettingsForm.on_model_changed()`).
- **Model Chains**: build and save a named, ordered sequence of
  captioning passes - each step its own model plus a full settings
  snapshot - and run them one after another over the same selected
  images from the `Model Chains...` button. Each step's model is loaded
  once and reused for every image before moving to the next step (the
  same model-cache reuse normal captioning already has), and a step's
  caption is actually applied (not just previewed), typically so a later
  step's prompt can reference an earlier step's output via the `{tags}`
  template variable - e.g. a WD Tagger pass for booru tags, then a
  Qwen3-VL pass prompted to write a natural-language caption from those
  tags (`taggui/auto_captioning/model_chains.py`,
  `taggui/dialogs/model_chain_dialog.py`).
- **UI scale setting**: a "UI scale (%)" spin box in Settings (50-300%,
  default 100) scales the entire GUI - fonts, icons, spacing, and widget
  sizes - via Qt's `QT_SCALE_FACTOR` environment variable, set from the
  saved setting before `QApplication` is constructed. Like the existing
  font size setting, it takes effect on restart, since Qt only reads this
  variable once, at startup (`taggui/run_gui.py:apply_ui_scale()`,
  `taggui/dialogs/settings_dialog.py`).
- **WD Tagger 8-bit quantization**, restricted to `MatMul` nodes only
  (quantizing the `ConvInteger` nodes in WD Tagger's ConvNeXt/EVA02
  backbones crashes some ONNX Runtime builds at inference time).

### Speed and throughput

These changes are about making large captioning runs finish in hours
instead of days; the reasoning behind each one is documented here because
the knobs interact and it's easy to undo one while fixing another.

- **Batch captioning** (`captioning_thread.py` + `Qwen3VL.get_batched_model_inputs()`):
  the Qwen3-VL family can now caption several images in a single
  `generate()` call (`Batch size` setting). At batch size 1, decode is
  memory-bandwidth-bound, so an N-image batch costs barely more than one
  image - this is the single biggest in-process speedup. Batching needs
  model-specific input padding (Qwen-style processors pad a list of
  texts/images natively), so it is opt-in per model class via
  `supports_batching = True`; other models silently fall back to one at a
  time. Batched calls pad with `padding_side='left'` so that slicing the
  generated tokens off by the padded input length is correct for every
  row. If a batched `generate()` hits CUDA OOM, the thread clears the
  cache and retries the same images one at a time instead of failing.
- **Prefetch pipeline**: while the GPU generates (or a remote request is
  in flight), a background thread prepares the next images' inputs (PIL
  decode + processor call, or base64 encoding for remote). This overlaps
  the CPU-heavy image work with GPU work; the pipeline window is small
  (`max(4, batch_size * 2)` preps ahead) so canceling doesn't decode a
  whole dataset's worth of images needlessly.
- **Remote API endpoint** (`Remote API (vLLM / SGLang)` pseudo-model,
  `auto_captioning/models/openai_compatible.py`): sends images as base64
  JPEG data URIs to an OpenAI-compatible server (vLLM, SGLang, ...) over
  `/chat/completions` instead of loading transformers in-process. This is
  the fastest way to caption very large datasets with Qwen3-VL-family
  models: the server does continuous batching, prefix caching,
  FlashAttention/FlashInfer, and fused MoE kernels (vLLM/SGLang don't run
  natively on Windows - use WSL2 or Docker). `Batch size` controls how
  many requests are in flight at once; the server-side model name is
  auto-detected from `/models` when left empty. Beam search is not
  supported (validated up front); think blocks are stripped
  unconditionally since there is no portable way to pass
  `enable_thinking=False` through the OpenAI API.
- **Pixel budget settings** (`Min/Max image size (MP)`, Qwen3-VL family
  only): the previously hardcoded `1280 * 28 * 28` max pixel budget is
  now a user control. Every 28x28 pixels is one visual token after the
  2x2 merge, so this directly sets prefill cost and VRAM per image (the
  UI shows the approximate visual-token count). Passed per call as
  `size={'shortest_edge': ..., 'longest_edge': ...}` - the
  `min_pixels`/`max_pixels` processor-call kwargs are deprecated in
  `transformers` (removal in 5.23) - so changing it doesn't reload the
  model.
- **Attention implementation picker** (`Default`/`SDPA`/`FlashAttention 2`):
  passed to `from_pretrained(attn_implementation=...)`. SageAttention
  stays a separate option because it works by monkey-patching
  `torch.nn.functional.scaled_dot_product_attention` (see below), which
  no `attn_implementation` value can express.
- **torch.compile (experimental)**: compiles the model (`dynamic=True`
  to avoid per-resolution recompiles). Incompatible with bitsandbytes
  4/8-bit (validated up front with a clear message) and with SageAttention
  (the monkey-patched SDPA function breaks tracing). The first generation
  after enabling it pays the compile cost; the model-load cache flag
  (`is_model_torch_compiled`) makes toggling it reload cleanly.
- **Quantized KV cache (fp8)**: passes `cache_implementation='quantized'`
  with `cache_config={'backend': 'fp8'}` to `generate()`. transformers'
  fp8 cache layer uses `torch.float8_e4m3fn` directly, so no extra
  package is needed (the `quanto`/`hqq` backends would require
  `optimum-quanto`/`hqq`). This roughly halves cache memory - which is
  what buys headroom for larger batch sizes - at a small per-token
  dequantization cost. Requires a `transformers` release that ships the
  `fp8` cache backend (validated up front).
- **Pre-quantized checkpoint passthrough**: when a checkpoint ships with
  its own `quantization_config` (GPTQ/AWQ/bitsandbytes/...), it is
  detected at load time, a note is printed, and `Load in 4-bit`/`Load in
  8-bit` is ignored rather than stacking bitsandbytes on top of an
  already-quantized model. Pre-quantized GPTQ/AWQ checkpoints use faster
  kernels (Marlin) than bnb NF4, so prefer them for 4-bit runs.
- **Warmup generation**: after each (re)load on GPU, a two-token text-only
  generation warms the decode-path CUDA kernels so the first image's
  timing (and the per-image averages derived from it) isn't inflated by
  one-time init. Best-effort: failures are swallowed since the first real
  image would surface the same problem anyway.
- **bad/forced words memoization**: `Discourage from caption`/`Include in
  caption` used to be re-tokenized for every image; they're now computed
  once per run.

### Long-run reliability and workflow

These changes assume the realistic failure model of an overnight
captioning run: some images will fail (corrupt files, transient CUDA
OOM, a server hiccup), the user will want to cancel mid-image, and the
run should survive all of that without losing hours of work.

- **Per-image error isolation**: any error during generation or image
  preparation is recorded against that image (printed to the console with
  the traceback text) and the run continues. At the end a failure summary
  lists what failed, and the widget offers to select the failed images in
  the image list - retrying them is then just pressing `Start
  Auto-Captioning` again (they're the selection). The run is only marked
  as errored (red alert) if *nothing* succeeded.
- **OOM resilience**: a batched call that hits CUDA OOM falls back to
  per-image captioning; a per-image OOM gets one retry at half the pixel
  budget (`prepare_reduced_inputs()`, Qwen3-VL family) before the image
  is reported as failed.
- **Skip-already-captioned**: a checkbox that filters out images that
  already have tags before the run starts (crash-resume in one click).
- **Cancel inside `generate()`**: a `StoppingCriteria`
  (`CancelCaptioningWhenRequested`) stops generation as soon as the
  cancel flag is set, instead of finishing the current image's whole
  token budget (minutes for a thinking model). The partial generation is
  discarded. Remote requests can't be canceled mid-flight (HTTP), so
  canceling a remote batch takes effect between requests.
- **ETA/throughput/VRAM in the progress bar**: each batch updates the
  bar's format with done/total, ETA, images/hour, and free/total VRAM
  (`torch.cuda.mem_get_info`), so the speed knobs can be tuned live.
- **Test run**: `Test run (0 = off)` captions the first N images, then
  pauses (the thread polls a flag) and shows a *non-modal* prompt -
  non-modal so the user can review the applied captions in the main
  window before choosing to continue or stop. Not supported for parallel
  multi-GPU runs (there's no single point where all workers pause
  cleanly); ignored there.
- **Compare Prompts dialog** (`dialogs/prompt_comparison_dialog.py`):
  captions the first N selected images with two prompts (A/B) side by
  side without applying captions (`Do not add to tags`, so no undo entry,
  no model changes). The thread is parented to the AutoCaptioner so the
  loaded model is reused; the AutoCaptioner's Start button is disabled
  while a comparison runs (they share the model cache).
- **Load presets** (`load_presets.py`): one-click recommended load/speed
  settings per model family (dense vs MoE Qwen3.5, JoyCaption,
  Florence-2, LLaVA, WD Tagger, remote endpoint) - dtype/quantization,
  thinking, batch size, pixel budget, KV cache. Starting points for a
  ~24-32 GB GPU, not verified optima.
- **Unload Model button**: frees the widget-cached model (`gc.collect()`
  + `torch.cuda.empty_cache()`). Disabled while captioning; a parallel
  run's per-GPU models are freed automatically when that run ends.
- **Two-GPU parallel captioning** (`Device` = `Multiple GPUs (parallel)`):
  one independent model instance per GPU (each with its own quantization/
  settings and its own `ModelCacheHolder` so the widget's single-model
  cache is untouched), the selected images split round-robin, one
  captioning pipeline per GPU. Progress/failures are shared via a lock
  and reported in the same progress bar/console. Requires >= 2 CUDA GPUs
  (otherwise it falls back to one device with a note). The widget-level
  unload button doesn't cover these (they're freed when the run ends).

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
