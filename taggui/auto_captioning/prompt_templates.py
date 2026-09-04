"""
Starting-point captioning prompts for common training targets and purposes.

These are general-purpose instructions for the auto-captioning model to
follow (not verified per-model training recipes) - a reasonable default to
insert into the `Prompt` box and adjust further, not a guarantee of optimal
results for any particular checkpoint or trainer.

Two broad styles are used:
- Models built around a CLIP-style text encoder with a short context window
  (SDXL) get concise, single-sentence instructions.
- Models built around a longer-context text encoder (Flux.1, Flux.2,
  Z-Image, Krea 2 Turbo) get detailed, multi-sentence instructions, since
  they can make use of (and are commonly trained on) longer captions.
`Anima` is treated as an anime/illustration-oriented target and uses
Danbooru-style comma-separated tags instead of prose, matching TagGUI's own
Danbooru tag autocomplete feature.

`PROMPT_TEMPLATES[model][purpose]` returns the prompt text; every model
defines the same set of purposes (`PROMPT_TEMPLATE_PURPOSES`).
"""

PROMPT_TEMPLATE_PURPOSES = ['General', 'Character', 'Style', 'Landscape']

PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    'SDXL': {
        'General': 'Describe this image in a single, clear sentence '
                   'covering the main subject, setting, and action.',
        'Character': 'Describe the character in this image in one concise '
                     'sentence, covering their appearance, clothing, pose, '
                     'and expression.',
        'Style': 'Describe the art style of this image in one concise '
                'sentence, covering the medium, rendering technique, color '
                'palette, and mood.',
        'Landscape': 'Describe this landscape image in one concise '
                     'sentence, covering the setting, time of day, '
                     'weather, and key environmental features.',
    },
    'Flux.1': {
        'General': 'Describe this image in a detailed paragraph, covering '
                   'the subject, setting, composition, lighting, and '
                   'mood.',
        'Character': 'Describe the character in this image in a detailed '
                     'paragraph, covering their physical appearance, '
                     'clothing, pose, expression, and any notable '
                     'accessories or props.',
        'Style': 'Describe the artistic style of this image in a detailed '
                'paragraph, covering the medium, technique, color '
                'palette, linework, shading, and overall mood.',
        'Landscape': 'Describe this landscape image in a detailed '
                     'paragraph, covering the terrain, vegetation, '
                     'weather, time of day, lighting, and atmosphere.',
    },
    'Flux.2': {
        'General': 'Describe this image in a detailed paragraph, covering '
                   'the subject, setting, composition, camera angle, '
                   'lighting, and mood.',
        'Character': 'Describe the character in this image in a detailed '
                     'paragraph, covering their appearance, clothing, '
                     'pose, expression, camera framing, and any notable '
                     'accessories or props.',
        'Style': 'Describe the artistic style of this image in a detailed '
                'paragraph, covering the medium, rendering technique, '
                'color palette, linework, shading, and overall mood.',
        'Landscape': 'Describe this landscape image in a detailed '
                     'paragraph, covering the terrain, vegetation, '
                     'weather, time of day, lighting, camera '
                     'perspective, and atmosphere.',
    },
    'Z-Image': {
        'General': 'Describe this image in a detailed paragraph, covering '
                   'the subject, setting, composition, lighting, and any '
                   'visible text.',
        'Character': 'Describe the character in this image in a detailed '
                     'paragraph, covering their appearance, clothing, '
                     'pose, expression, and any notable accessories or '
                     'props.',
        'Style': 'Describe the artistic style of this image in a detailed '
                'paragraph, covering the medium, technique, color '
                'palette, and overall mood.',
        'Landscape': 'Describe this landscape image in a detailed '
                     'paragraph, covering the terrain, vegetation, '
                     'weather, time of day, and lighting.',
    },
    'Krea 2 Turbo': {
        'General': 'Describe this image in a detailed, photographic '
                   'paragraph, covering the subject, setting, lighting, '
                   'and camera perspective.',
        'Character': 'Describe the person in this image in a detailed, '
                     'photographic paragraph, covering their appearance, '
                     'clothing, pose, expression, and the lighting and '
                     'camera angle.',
        'Style': 'Describe the visual style of this image in a detailed '
                'paragraph, covering the photographic technique, '
                'lighting setup, color grading, and overall mood.',
        'Landscape': 'Describe this landscape image in a detailed, '
                     'photographic paragraph, covering the terrain, '
                     'weather, time of day, and lighting conditions.',
    },
    'Anima': {
        'General': 'List Danbooru-style tags describing this image, '
                   'separated by commas, covering the subject, setting, '
                   'and notable visual elements.',
        'Character': 'List Danbooru-style tags describing the character '
                     'in this image, separated by commas, covering hair '
                     'color and style, eye color, clothing, pose, and '
                     'expression.',
        'Style': 'List Danbooru-style tags describing the art style of '
                'this image, separated by commas, covering the medium, '
                'shading technique, and color palette.',
        'Landscape': 'List Danbooru-style tags describing this landscape '
                     'image, separated by commas, covering the setting, '
                     'time of day, weather, and background elements.',
    },
}
