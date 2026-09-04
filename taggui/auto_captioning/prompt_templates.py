"""
Starting-point captioning prompts for common training targets and purposes.

These are instructions for the auto-captioning model to follow when writing
the caption/tag file for each image - general guidance drawn from official
docs and widely-used community LoRA/fine-tuning guides for each target,
condensed into a single instruction. They are a reasonable default to
insert into the `Prompt` box and adjust further, not a guarantee of optimal
results for any particular checkpoint or trainer.

Two things vary by target model, based on research into each one's actual
captioning conventions (see notes on each model below):

- Caption style and length. SDXL's CLIP text encoders truncate at around
  75 tokens, so its captions stay short. Flux.1, Flux.2, Z-Image, and
  Krea 2 use longer-context text encoders (T5/Mistral/Qwen3-VL-style) and
  are commonly trained on detailed, full-sentence natural language
  instead. Anima is trained on Danbooru-style tags, natural language, and
  mixes of both, so its templates default to tags to match TagGUI's own
  Danbooru autocomplete support.

- Whether to describe style. For a Style-purpose caption on the
  natural-language-caption models (SDXL, Flux.1, Flux.2, Z-Image,
  Krea 2), the templates deliberately ask for content only, with no style
  words. This isn't an oversight: the common LoRA-training convention
  (and Black Forest Labs' own published guidance for Flux.2 style
  training) is that if a style is described consistently in every
  caption, the model can't tell it apart from "how things just look" and
  won't isolate it as a learnable, controllable concept - it has to be
  the one thing that's consistently *present* in the images but
  consistently *absent* from the text. Anima is the exception: Danbooru's
  tagging convention treats medium/technique/artist tags as normal,
  directly controllable tags (not something to hide from the caption), so
  its Style template asks for exactly those tags instead.

`PROMPT_TEMPLATES[model][purpose]` returns the prompt text; every model
defines the same set of purposes (`PROMPT_TEMPLATE_PURPOSES`).
"""

PROMPT_TEMPLATE_PURPOSES = ['General', 'Character', 'Style', 'Landscape']

PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    'SDXL': {
        'General': 'Describe this image in a single, clear sentence '
                   '(no more than about 60 words, since SDXL\'s text '
                   'encoders truncate long captions), covering the main '
                   'subject, setting, and action.',
        'Character': 'Describe the character in this image in one '
                     'concise sentence. Focus on things that vary '
                     'between images of them, such as clothing, pose, '
                     'expression, and setting. Do not describe fixed '
                     'physical traits like hair color, eye color, or '
                     'face shape, since those should be learned as part '
                     'of the character rather than treated as variable.',
        'Style': 'Describe only the content of this image in one '
                'concise sentence: the subject, setting, and '
                'composition. Do not mention the art style, medium, or '
                'rendering technique, so that the visual style is '
                'learned implicitly from the images instead of tied to '
                'specific caption words.',
        'Landscape': 'Describe this landscape image in one concise '
                     'sentence, covering the setting, time of day, '
                     'weather, and key environmental features.',
    },
    'Flux.1': {
        'General': 'Describe this image the way you would prompt Flux '
                   'to generate it: one or two natural-language '
                   'sentences (under about 75 tokens) covering the '
                   'subject, setting, composition, and lighting.',
        'Character': 'Describe the character in this image in one or '
                     'two natural-language sentences. Focus on what '
                     'varies between images of them - clothing, '
                     'accessories, pose, expression, and setting - and '
                     'leave out fixed identity traits like hair color, '
                     'eye color, and face shape, since those should be '
                     'learned as part of the character rather than '
                     'treated as variable.',
        'Style': 'Describe only the content of this image in one or '
                'two natural-language sentences: the subject, setting, '
                'and composition. Do not mention the art style, '
                'medium, or rendering technique, so the visual style '
                'is learned implicitly from the images rather than '
                'tied to specific caption words.',
        'Landscape': 'Describe this landscape image in one or two '
                     'natural-language sentences, covering the '
                     'terrain, vegetation, weather, time of day, and '
                     'lighting.',
    },
    'Flux.2': {
        'General': 'Describe this image in a detailed natural-language '
                   'paragraph of about 40-100 words, covering the '
                   'subject, setting, composition, camera angle, and '
                   'lighting.',
        'Character': 'Describe the character in this image in a '
                     'detailed natural-language paragraph of about '
                     '40-100 words. Focus on what varies between '
                     'images of them - clothing, accessories, pose, '
                     'expression, camera framing, and setting - and '
                     'leave out fixed identity traits like hair color, '
                     'eye color, and face shape, since those should be '
                     'learned as part of the character rather than '
                     'treated as variable.',
        'Style': 'Describe only the content of this image in a '
                'detailed natural-language paragraph of about 40-100 '
                'words: the subject, setting, and composition. Do not '
                'mention the art style, medium, rendering technique, '
                'or quality descriptors, so the visual style is '
                'learned implicitly from the images rather than tied '
                'to specific caption words.',
        'Landscape': 'Describe this landscape image in a detailed '
                     'natural-language paragraph of about 40-100 '
                     'words, covering the terrain, vegetation, '
                     'weather, time of day, lighting, and camera '
                     'perspective.',
    },
    'Z-Image': {
        'General': 'Describe this image in a detailed natural-language '
                   'paragraph, in this order: the subject (who or '
                   'what, with age/clothing/materials if applicable), '
                   'the scene (where and when), the composition '
                   '(camera angle and framing), and the lighting '
                   '(direction, color, time of day).',
        'Character': 'Describe the character in this image in a '
                     'detailed natural-language paragraph, covering '
                     'their clothing, pose, expression, the scene '
                     'around them, camera framing, and lighting. Leave '
                     'out fixed identity traits like hair color, eye '
                     'color, and face shape, since those should be '
                     'learned as part of the character rather than '
                     'treated as variable.',
        'Style': 'Describe only the content of this image in a '
                'detailed natural-language paragraph: the subject, '
                'scene, composition, and lighting. Do not mention the '
                'art style, medium, or rendering technique, so the '
                'visual style is learned implicitly from the images '
                'rather than tied to specific caption words.',
        'Landscape': 'Describe this landscape image in a detailed '
                     'natural-language paragraph, covering the '
                     'terrain and scene, camera angle and framing, '
                     'and the lighting (direction, color, time of '
                     'day).',
    },
    'Krea 2 Turbo': {
        'General': 'Describe this image in a detailed natural-language '
                   'paragraph, covering the subject, setting, '
                   'lighting, and camera perspective. Do not use '
                   'comma-separated tags.',
        'Character': 'Describe the person in this image in a detailed '
                     'natural-language paragraph. Focus on what '
                     'varies between images of them - clothing, pose, '
                     'expression, lighting, and camera angle - and '
                     'leave out fixed identity traits like hair '
                     'color, eye color, and face shape, since those '
                     'should be learned as part of the character '
                     'rather than treated as variable.',
        'Style': 'Describe only the content of this image in a '
                'detailed natural-language paragraph: the subject, '
                'setting, and composition. Do not mention the '
                'photographic style, color grading, or rendering '
                'technique, so the visual style is learned implicitly '
                'from the images rather than tied to specific caption '
                'words.',
        'Landscape': 'Describe this landscape image in a detailed '
                     'natural-language paragraph, covering the '
                     'terrain, weather, time of day, and lighting '
                     'conditions.',
    },
    'Anima': {
        'General': 'List Danbooru-style tags describing this image, '
                   'separated by commas, in this order: rating/quality '
                   'tags, subject count (such as 1girl or 1boy), '
                   'character name if recognizable, series name if '
                   'recognizable, and general tags for the setting and '
                   'notable visual elements.',
        'Character': 'List Danbooru-style tags describing the '
                     'character in this image, separated by commas: '
                     'subject count (such as 1girl or 1boy), character '
                     'name and series if recognizable, then hair '
                     'color and style, eye color, clothing, pose, and '
                     'expression.',
        'Style': 'List Danbooru-style tags describing the art style of '
                'this image, separated by commas, covering the medium '
                '(such as watercolor, sketch, or 3d), the artist if '
                'recognizable, the shading technique, and the color '
                'palette.',
        'Landscape': 'List Danbooru-style tags describing this '
                     'landscape image, separated by commas, covering '
                     'the setting, time of day, weather, and '
                     'background elements.',
    },
}
