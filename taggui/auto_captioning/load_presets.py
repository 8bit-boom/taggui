"""
Recommended load/speed settings per captioning model family, applied with
one click from the Auto-Captioner (`Apply` next to the preset picker).
These are researched starting points for a single ~24-32 GB NVIDIA GPU,
not verified optima for every card: a smaller GPU may need 4-bit where a
preset says bf16, and a bigger one can raise the batch size. Only the keys
present in a preset are applied, so a preset can't clobber settings it
doesn't know about.
"""

LOAD_PRESETS: dict[str, dict] = {
    'Qwen3-VL / Qwen3.5 dense (8B-class)': {
        # An 8B dense model in bf16 (~17 GB) captions faster than its
        # bitsandbytes 4-bit version and still leaves batch headroom on a
        # 24 GB card.
        'load_in_4_bit': False,
        'load_in_8_bit': False,
        'cpu_offload': False,
        'sage_attention': False,
        'disable_thinking': True,
        'attention_implementation': 'Default',
        'torch_compile': False,
        'quantize_kv_cache': False,
        'batch_size': 4,
        'min_image_megapixels': 0.2,
        'max_image_megapixels': 1.0
    },
    'Qwen3.5 MoE (A3B)': {
        # A 35B-A3B checkpoint only fits unquantized on ~80 GB cards, so
        # 4-bit it is; the fp8 KV cache buys the headroom for a larger
        # batch. MoE models are bottlenecked by expert-gather bandwidth,
        # which batching amortizes well.
        'load_in_4_bit': True,
        'load_in_8_bit': False,
        'cpu_offload': False,
        'sage_attention': False,
        'disable_thinking': True,
        'attention_implementation': 'Default',
        'torch_compile': False,
        'quantize_kv_cache': True,
        'batch_size': 4,
        'min_image_megapixels': 0.2,
        'max_image_megapixels': 1.0
    },
    'JoyCaption': {
        'load_in_4_bit': True,
        'load_in_8_bit': False,
        'cpu_offload': False,
        'sage_attention': False,
        'attention_implementation': 'Default',
        'torch_compile': False,
        'quantize_kv_cache': False,
        'batch_size': 1
    },
    'Florence-2': {
        'load_in_4_bit': False,
        'load_in_8_bit': False,
        'cpu_offload': False,
        'sage_attention': False,
        'attention_implementation': 'Default',
        'torch_compile': False,
        'quantize_kv_cache': False,
        'batch_size': 8
    },
    'LLaVA / Moondream': {
        'load_in_4_bit': True,
        'load_in_8_bit': False,
        'cpu_offload': False,
        'sage_attention': False,
        'attention_implementation': 'Default',
        'torch_compile': False,
        'quantize_kv_cache': False,
        'batch_size': 1
    },
    'WD Tagger': {
        'wd_tagger_load_in_8_bit': True,
        'batch_size': 1
    },
    'Remote API (vLLM / SGLang)': {
        # The server does its own batching; the client-side batch size is
        # just how many requests are in flight, so a high number is cheap.
        'batch_size': 8
    }
}
