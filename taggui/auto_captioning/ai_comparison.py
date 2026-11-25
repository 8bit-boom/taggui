"""
Core logic for the one-button AI comparison of command prompts and settings.

The flow (orchestrated by `AIComparisonDialog`):
  1. Build candidate configurations = prompt candidates x settings candidates.
  2. Caption a few sample images with each candidate (captions not applied).
  3. Ask a judge LLM (a local OpenAI-compatible server, e.g. Ollama / vLLM /
     llama.cpp serving Qwen3.8-27B or Gemma 4) to score and rank the candidates.
  4. Present a ranked report; the winning prompt/settings can be applied.

The functions here are pure (except `judge_captions`, which does the HTTP
call) so they can be unit-tested without a model or a GUI.
"""

import json
import re

from auto_captioning.models.openai_compatible import (
    REQUEST_TIMEOUT_SECONDS, THINK_BLOCK_PATTERN, get_json, post_json)

# Cap the cartesian product so a stray selection can't launch hundreds of
# captioning runs from a single button press.
MAX_CANDIDATES = 24


def candidate_label(index: int) -> str:
    """A, B, ... Z, AA, AB, ... for candidate indices."""
    label = ''
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord('A') + remainder) + label
    return label


def build_candidates(prompt_candidates: list[str],
                     settings_candidates: list[dict],
                     base_settings: dict) -> list[dict]:
    """
    Cartesian product of prompts x settings, each a full captioning-settings
    snapshot ready for `CaptioningThread`.

    `prompt_candidates`: list of prompt strings (empty/blank ones skipped).
    `settings_candidates`: list of {'name': str, 'generation_parameters': dict};
        an empty list means "use the current settings only".
    `base_settings`: the form's current settings snapshot (model, device, etc.).
    """
    prompts = [prompt for prompt in prompt_candidates if prompt.strip()]
    if not prompts:
        return []
    if not settings_candidates:
        settings_candidates = [{'name': 'Current settings',
                                'generation_parameters':
                                    base_settings.get('generation_parameters',
                                                      {})}]
    candidates = []
    for prompt in prompts:
        for settings in settings_candidates:
            if len(candidates) >= MAX_CANDIDATES:
                return candidates
            index = len(candidates)
            snapshot = dict(base_settings)
            snapshot['prompt'] = prompt
            snapshot['generation_parameters'] = dict(
                settings.get('generation_parameters', {}))
            # Comparison captions are never applied to the image list.
            snapshot['caption_position'] = 'Do not add to tags'
            snapshot['skip_already_captioned'] = False
            snapshot['test_run_image_count'] = 0
            prompt_snippet = prompt.strip().replace('\n', ' ')
            if len(prompt_snippet) > 50:
                prompt_snippet = prompt_snippet[:50] + '...'
            candidates.append({
                'label': candidate_label(index),
                'prompt': prompt,
                'settings_name': settings.get('name', 'Current settings'),
                'settings': snapshot,
                'description': f'"{prompt_snippet}" [{settings.get("name", "Current settings")}]',
            })
    return candidates


def build_judge_prompt(candidates: list[dict], results: list[dict],
                       target_description: str = '') -> str:
    """
    Build the text-only judging prompt. `results` is a list of
    {'image': name, 'captions': {label: caption}} in candidate order.
    The judge is asked to reply with a single JSON object.
    """
    labels = [candidate['label'] for candidate in candidates]
    lines = [
        'You are an expert judge of image captions used to train generative '
        'AI models. Several candidate captioning configurations (different '
        'command prompts and generation settings) were each run on the same '
        'sample images. Compare the captions they produced and decide which '
        'configuration is best.',
        '',
        'Judge caption quality on: accuracy to the image, useful visual '
        'detail, clarity, and suitability for training (no refusals, '
        'preambles, meta-commentary, or repeated filler). Penalize empty, '
        'truncated, or off-topic captions.',
    ]
    if target_description:
        lines += ['', f'The captions are intended for: {target_description}.']
    lines += ['', 'The candidates are:']
    for candidate in candidates:
        lines.append(f'  {candidate["label"]}: {candidate["description"]}')
    lines += ['', 'Captions per image:']
    for result in results:
        lines.append(f'Image: {result["image"]}')
        for label in labels:
            caption = result['captions'].get(label, '') or '(empty)'
            lines.append(f'  {label}: {caption}')
        lines.append('')
    lines += [
        'Reply with ONLY a JSON object (no markdown, no commentary) of this '
        'exact shape:',
        '{',
        '  "per_image": [{"image": "<name>", "scores": {"<label>": <0-10>}, '
        '"best": "<label>", "comment": "<short>"}],',
        '  "overall": {"scores": {"<label>": <0-10 average>}, '
        '"ranking": ["<best label>", "..."], "best": "<label>", '
        '"reasoning": "<why the winner is best, 2-4 sentences>"}',
        '}',
        f'Use exactly these candidate labels: {", ".join(labels)}.',
    ]
    return '\n'.join(lines)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response (tolerates fences)."""
    text = THINK_BLOCK_PATTERN.sub('', text).strip()
    # Strip markdown code fences if present.
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find('{')
    if start == -1:
        return None
    # Try progressively shorter substrings to tolerate trailing commentary.
    end = text.rfind('}')
    while end > start:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            end = text.rfind('}', start, end)
    return None


def parse_judge_response(text: str, labels: list[str]) -> dict:
    """
    Normalize a judge response into {'ranking', 'scores', 'best', 'reasoning',
    'per_image', 'valid'}. Defensive: a malformed response yields a fallback
    ranking (candidate order) with valid=False rather than crashing.
    """
    fallback = {
        'ranking': list(labels),
        'scores': {label: 0.0 for label in labels},
        'best': labels[0] if labels else None,
        'reasoning': 'The judge did not return a usable ranking.',
        'per_image': [],
        'valid': False,
    }
    data = _extract_json(text)
    if not isinstance(data, dict):
        return fallback
    overall = data.get('overall') or {}
    if not isinstance(overall, dict):
        overall = {}

    raw_scores = overall.get('scores') or {}
    scores = {}
    for label in labels:
        value = raw_scores.get(label) if isinstance(raw_scores, dict) else None
        try:
            scores[label] = float(value)
        except (TypeError, ValueError):
            scores[label] = 0.0

    ranking = overall.get('ranking')
    if isinstance(ranking, list):
        ranking = [label for label in ranking if label in labels]
        # Append any labels the judge omitted, keeping a complete ranking.
        ranking += [label for label in labels if label not in ranking]
    else:
        ranking = sorted(labels, key=lambda label: scores[label], reverse=True)
    if not ranking:
        ranking = list(labels)

    best = overall.get('best')
    if best not in labels:
        best = ranking[0] if ranking else None
    reasoning = overall.get('reasoning') or ''
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    per_image = data.get('per_image')
    if not isinstance(per_image, list):
        per_image = []
    return {
        'ranking': ranking,
        'scores': scores,
        'best': best,
        'reasoning': reasoning.strip(),
        'per_image': per_image,
        'valid': True,
    }


def resolve_judge_model(base_url: str, api_key: str | None,
                        model: str | None) -> str | None:
    """Use the given model name, else auto-detect the first served model."""
    if model and model.strip():
        return model.strip()
    try:
        response = get_json(f'{base_url.rstrip("/")}/models', api_key,
                            REQUEST_TIMEOUT_SECONDS)
        return response['data'][0]['id']
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def judge_captions(base_url: str, api_key: str | None, model: str | None,
                   judge_prompt: str) -> str:
    """
    Send the judging prompt to a local OpenAI-compatible server and return the
    raw response text. Raises on connection/HTTP errors (the dialog reports
    them). Greedy decoding for a deterministic, parseable verdict.
    """
    resolved = resolve_judge_model(base_url, api_key, model)
    if not resolved:
        raise RuntimeError(
            f'Could not reach a judge model at {base_url}. Start a local '
            f'OpenAI-compatible server (Ollama / vLLM / llama.cpp) serving a '
            f'capable LLM (e.g. Qwen3.8-27B or Gemma 4), or set the judge '
            f'model name explicitly.')
    payload = {
        'model': resolved,
        'messages': [{'role': 'user', 'content': judge_prompt}],
        'temperature': 0,
        'max_tokens': 2000,
    }
    response = post_json(f'{base_url.rstrip("/")}/chat/completions', payload,
                         api_key, REQUEST_TIMEOUT_SECONDS)
    try:
        return response['choices'][0]['message']['content'] or ''
    except (KeyError, IndexError, TypeError) as exception:
        raise RuntimeError(
            f'The judge server returned an unexpected response: {response}'
        ) from exception
