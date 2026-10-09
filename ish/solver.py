"""Wei, Hu, Leung and Zhang's improved skyline heuristic BestFitPack (2017).

The ISH (COR 80:113-127) keeps Burke's best-fit framework — take the most
bottom-left skyline segment, select the best-fitting rectangle, place it —
but replaces the widest-first selection with the fitness number: the number
of rectangle sides that exactly match the touched skyline walls (bottom if
r.w == s.w, left if r.h == s.lh, right if r.h == s.rh), worth 0-3. Selection
checks fitness 3, then 2, then 1, then the first fitting rectangle, breaking
ties by the earliest index in the input sequence. The rectangle is placed at
the segment corner with the better fitness, preferring the tallest neighbour
on ties (left corner when equal). When nothing fits, the segment is raised
to its lower neighbour. Fixed orientation only, as in the paper.

The paper's O(n log n) implementation (heap + auxiliary sorted sequences and
a segment tree) is not reproduced; the selection here is a direct scan with
identical outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from burke_bf.model import Item, Placement


@dataclass(frozen=True, slots=True)
class _Segment:
    x: int
    y: int
    w: int

    @property
    def right(self) -> int:
        return self.x + self.w


def _merge(segments: list[_Segment]) -> list[_Segment]:
    merged: list[_Segment] = []
    for seg in sorted(segments, key=lambda segment: segment.x):
        if merged and merged[-1].right == seg.x and merged[-1].y == seg.y:
            last = merged[-1]
            merged[-1] = _Segment(last.x, last.y, last.w + seg.w)
        else:
            merged.append(seg)
    return merged


def best_fit_pack(
    sequence: list[Item], strip_width: int
) -> tuple[list[Placement], int]:
    """Pack the sequence with ISH's BestFitPack; return (placements, height)."""

    remaining = list(enumerate(sequence))
    segments: list[_Segment] = [_Segment(0, 0, strip_width)]
    placements: list[Placement] = []

    while remaining:
        j = min(
            range(len(segments)),
            key=lambda index: (segments[index].y, segments[index].x),
        )
        seg = segments[j]
        lh = segments[j - 1].y - seg.y if j > 0 else inf
        rh = segments[j + 1].y - seg.y if j < len(segments) - 1 else inf

        chosen = None  # (fitness, r.i, item)
        if lh == rh:
            for index, item in remaining:
                if item.width == seg.w and item.height == lh:
                    chosen = (3, index, item)
                    break
        if chosen is None:
            for index, item in remaining:
                if item.width == seg.w and item.height in (lh, rh):
                    chosen = (2, index, item)
                    break
        if chosen is None:
            best = None
            for index, item in remaining:
                if (
                    item.width == seg.w
                    or (item.height == lh and item.width <= seg.w)
                    or (item.height == rh and item.width <= seg.w)
                ):
                    if best is None or index < best[1]:
                        best = (1, index, item)
            chosen = best
        if chosen is None:
            for index, item in remaining:
                if item.width <= seg.w:
                    chosen = (0, index, item)
                    break

        if chosen is None:
            left_y = segments[j - 1].y if j > 0 else inf
            right_y = segments[j + 1].y if j < len(segments) - 1 else inf
            new_y = min(left_y, right_y)
            if new_y == inf:
                raise RuntimeError("no rectangle fits an empty full-width strip")
            segments = _merge(
                segments[:j]
                + [_Segment(seg.x, int(new_y), seg.w)]
                + segments[j + 1 :]
            )
            continue

        _fitness, index, item = chosen
        fit_left = (item.width == seg.w) + (item.height == lh) + (
            item.width == seg.w and item.height == rh
        )
        fit_right = (item.width == seg.w) + (item.height == rh) + (
            item.width == seg.w and item.height == lh
        )
        if fit_left != fit_right:
            at_left = fit_left > fit_right
        else:
            at_left = lh >= rh  # tallest neighbour, left corner on ties
        rx = seg.x if at_left else seg.right - item.width

        placements.append(
            Placement(
                item_id=item.item_id,
                x=rx,
                y=seg.y,
                width=item.width,
                height=item.height,
                original_width=item.width,
                original_height=item.height,
            )
        )
        remaining = [(i, other) for i, other in remaining if i != index]

        new_segments: list[_Segment] = []
        for other in segments:
            if other.right <= rx or other.x >= rx + item.width:
                new_segments.append(other)
                continue
            if other.x < rx:
                new_segments.append(_Segment(other.x, other.y, rx - other.x))
            if other.right > rx + item.width:
                new_segments.append(
                    _Segment(rx + item.width, other.y, other.right - (rx + item.width))
                )
        new_segments.append(_Segment(rx, seg.y + item.height, item.width))
        segments = _merge(new_segments)

    height = max(placement.top for placement in placements)
    return placements, height
