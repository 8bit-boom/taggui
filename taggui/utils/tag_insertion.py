"""
Precise tag insertion shared by manual tag entry and auto-captioning.

Both paths used to only append (manual) or offer a few fixed caption
positions (auto). This module gives them one insertion engine with explicit
modes: at the top, at the bottom, at an arbitrary position *between* tags,
and - for manual entry relative to the tag list's current selection -
before / after / replacing the selected tags.

`insert_tags` never mutates its inputs; it returns a new list.
"""

from enum import Enum


class TagInsertMode(str, Enum):
    BOTTOM = 'At bottom (after last tag)'
    TOP = 'At top (before first tag)'
    AT_INDEX = 'At position (between tags)'
    BEFORE_SELECTED = 'Before selected tag'
    AFTER_SELECTED = 'After selected tag'
    REPLACE_SELECTED = 'Replace selected tag'


def _clamp_index(index: int, length: int) -> int:
    """Clamp an insertion index into [0, length] (0 = top, length = bottom)."""
    if index < 0:
        return 0
    if index > length:
        return length
    return index


def insert_tags(existing: list[str], new_tags: list[str],
                mode: TagInsertMode, index: int = 0,
                selected_rows: list[int] | None = None) -> list[str]:
    """
    Return `existing` with `new_tags` inserted according to `mode`.

    - TOP / BOTTOM: prepend / append.
    - AT_INDEX: insert at `index` (clamped to [0, len(existing)]), i.e. between
      tags; index 0 is the top and len(existing) is the bottom.
    - BEFORE_SELECTED / AFTER_SELECTED: insert just before the first selected
      row / just after the last selected row. With no selection these fall
      back to BOTTOM (there is no meaningful anchor).
    - REPLACE_SELECTED: remove the selected rows and put `new_tags` where the
      first selected row was. With no selection this falls back to BOTTOM.

    `selected_rows` are row indices into `existing`; out-of-range rows are
    ignored. Inputs are not mutated.
    """
    existing = list(existing)
    new_tags = list(new_tags)
    if not new_tags:
        return existing
    length = len(existing)

    if mode == TagInsertMode.TOP:
        return new_tags + existing
    if mode == TagInsertMode.BOTTOM:
        return existing + new_tags
    if mode == TagInsertMode.AT_INDEX:
        position = _clamp_index(index, length)
        return existing[:position] + new_tags + existing[position:]

    # Selection-relative modes.
    rows = sorted({row for row in (selected_rows or [])
                   if 0 <= row < length})
    if not rows:
        # No usable anchor: append rather than guess a position.
        return existing + new_tags
    first, last = rows[0], rows[-1]
    if mode == TagInsertMode.BEFORE_SELECTED:
        return existing[:first] + new_tags + existing[first:]
    if mode == TagInsertMode.AFTER_SELECTED:
        return existing[:last + 1] + new_tags + existing[last + 1:]
    if mode == TagInsertMode.REPLACE_SELECTED:
        row_set = set(rows)
        kept = [tag for row, tag in enumerate(existing) if row not in row_set]
        # Insert at the position of the first removed row within the kept list
        # (rows before `first` are all kept, so that count is `first`).
        return kept[:first] + new_tags + kept[first:]
    return existing + new_tags
