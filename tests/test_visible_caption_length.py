"""
Tests for `VisibleCaptionLengthState` and the two generation-control
classes built on it (`SuppressEosUntilVisibleMinLength`,
`VisibleCaptionMaxLengthCriteria`). These reinterpret a thinking model's
`min_new_tokens`/`max_new_tokens` as bounds on the caption actually shown
to the user (the text after the model's closing `</think>` tag) instead of
on the whole generation including hidden reasoning - see the long comment
on `VisibleCaptionLengthState` in `auto_captioning_model.py` for why. No
GPU or real model is needed: a fake tokenizer stands in, decoding each
token id to a fixed string so the tests can place `</think>` at an exact
position.
"""

import torch

from auto_captioning.auto_captioning_model import (
    THINKING_TOKEN_BUDGET, SuppressEosUntilVisibleMinLength,
    VisibleCaptionLengthState, VisibleCaptionMaxLengthCriteria)

PROMPT_LENGTH = 3
# Token id 1 decodes as the closing tag itself; every other id decodes as
# one filler character, so a row's decoded text is predictable from its
# token ids alone.
THINK_CLOSE_TOKEN_ID = 1


class FakeTokenizer:
    @staticmethod
    def decode(ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        return ''.join(
            '</think>' if token_id == THINK_CLOSE_TOKEN_ID else 'x'
            for token_id in ids.tolist())


def make_input_ids(*rows: list[int]) -> torch.Tensor:
    prompt = [0] * PROMPT_LENGTH
    return torch.tensor([prompt + row for row in rows])


def test_visible_token_count_is_zero_while_still_thinking():
    state = VisibleCaptionLengthState(FakeTokenizer(), PROMPT_LENGTH)
    input_ids = make_input_ids([0, 0, 0, 0])
    assert state.visible_token_count(input_ids, row=0) == 0


def test_visible_token_count_starts_after_closing_tag():
    state = VisibleCaptionLengthState(FakeTokenizer(), PROMPT_LENGTH)
    # Step 1: still thinking, no closing tag yet.
    input_ids = make_input_ids([0, 0])
    assert state.visible_token_count(input_ids, row=0) == 0
    # Step 2: the closing tag appears; visible count starts from here (0
    # for this call, since it was only just detected).
    input_ids = make_input_ids([0, 0, THINK_CLOSE_TOKEN_ID])
    assert state.visible_token_count(input_ids, row=0) == 0
    # Step 3-4: two more tokens generated after the tag was seen.
    input_ids = make_input_ids([0, 0, THINK_CLOSE_TOKEN_ID, 0])
    assert state.visible_token_count(input_ids, row=0) == 1
    input_ids = make_input_ids([0, 0, THINK_CLOSE_TOKEN_ID, 0, 0])
    assert state.visible_token_count(input_ids, row=0) == 2


def test_visible_token_count_tracks_rows_independently():
    state = VisibleCaptionLengthState(FakeTokenizer(), PROMPT_LENGTH)
    # Row 0 closes the tag now (0 visible so far - detection itself doesn't
    # retroactively count); row 1 is still thinking.
    input_ids = make_input_ids([THINK_CLOSE_TOKEN_ID, 0, 0], [0, 0, 0])
    assert state.visible_token_count(input_ids, row=0) == 0
    assert state.visible_token_count(input_ids, row=1) == 0
    # One step later: row 0 has 2 new tokens since its tag closed; row 1
    # still hasn't closed its own.
    input_ids = make_input_ids([THINK_CLOSE_TOKEN_ID, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0])
    assert state.visible_token_count(input_ids, row=0) == 2
    assert state.visible_token_count(input_ids, row=1) == 0


def test_visible_token_count_treats_whole_generation_as_visible_past_budget():
    state = VisibleCaptionLengthState(FakeTokenizer(), PROMPT_LENGTH)
    never_closes = [0] * THINKING_TOKEN_BUDGET
    input_ids = make_input_ids(never_closes)
    assert (state.visible_token_count(input_ids, row=0)
            == THINKING_TOKEN_BUDGET)


def test_suppress_eos_until_visible_min_length():
    state = VisibleCaptionLengthState(FakeTokenizer(), PROMPT_LENGTH)
    processor = SuppressEosUntilVisibleMinLength(
        state, min_new_tokens=3, eos_token_ids=[42])
    # Still thinking: EOS must stay suppressed.
    input_ids = make_input_ids([0, 0])
    scores = torch.zeros((1, 100))
    scores = processor(input_ids, scores)
    assert scores[0, 42] == float('-inf')
    # Past the closing tag but short of the 3-token visible minimum.
    input_ids = make_input_ids([THINK_CLOSE_TOKEN_ID, 0])
    scores = torch.zeros((1, 100))
    scores = processor(input_ids, scores)
    assert scores[0, 42] == float('-inf')
    # The visible minimum is reached: EOS is no longer suppressed. (The
    # tag closed on the previous call, at 2 generated tokens; 3 more
    # generated tokens since then means 3 visible tokens now.)
    input_ids = make_input_ids([THINK_CLOSE_TOKEN_ID, 0, 0, 0, 0])
    scores = torch.zeros((1, 100))
    scores = processor(input_ids, scores)
    assert scores[0, 42] == 0


def test_visible_caption_max_length_criteria_stops_each_row_independently():
    state = VisibleCaptionLengthState(FakeTokenizer(), PROMPT_LENGTH)
    criteria = VisibleCaptionMaxLengthCriteria(state, max_new_tokens=2)
    # Row 0's tag closes now; row 1 is still thinking. Neither is done yet.
    input_ids = make_input_ids([THINK_CLOSE_TOKEN_ID, 0, 0], [0, 0, 0])
    is_done = criteria(input_ids, scores=None)
    assert is_done.tolist() == [False, False]
    # One step later: row 0 has reached its 2-token visible maximum; row 1
    # still hasn't closed its own tag.
    input_ids = make_input_ids([THINK_CLOSE_TOKEN_ID, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0])
    is_done = criteria(input_ids, scores=None)
    assert is_done.tolist() == [True, False]
