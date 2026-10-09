"""Leung and Zhang's fast layer-based heuristic (FH, ESWA 38:13032-13042).

FH places rectangles layer by layer. Each layer starts with a reference
rectangle — the first unplaced one in the current ordering, laid with its
long edge along the reference line — and stacks further rectangles sharing
one side length with it until the reference line exceeds the lower bound
LB = ceil(total area / W). The space under the reference line is then filled
from the lowest available position upwards: for every free space S (walls
h1, h2) each unplaced rectangle is scored by a fitness value with one point
per matched condition — [width == S.width], [height == taller wall height],
[height reaches the reference line] — and the best-scoring one is placed
next to the taller wall, ties broken by the front of the ordering.

Re-examination note (2026-08): a close reading of the paper's Fig. 3 shows
the corner positions are the two bottom corners (width fill) and the taller
wall's top, i.e. the figure-faithful rule is 2 x [width == S.width] +
1 x [height == taller wall]; it reproduces all eight fitness levels of the
figure and matches 21/29 instances, all eight mismatches exactly +1. The
three-component rule implemented here does not reproduce Fig. 3(1.1)'s
value 3 in the general geometry (no reference-line corner exists there) but
scores one more exact match (22/29) with mixed deviations, so it is kept
as the better-agreeing reading and the tension is documented in the paper.
Spaces too small for any
unplaced rectangle are raised to the lower wall. Finally, a first-improvement
full swap pass over the ordering (swap every pair once, keep the swap when
the repacked height improves) forms the FH algorithm. The problem is the RF
subtype (rotation allowed).
"""

from __future__ import annotations

from math import ceil

from burke_bf.model import Item, Placement


def _heuristic_placing(
    sequence: list[Item], strip_width: int, lower_bound: int
) -> tuple[list[Placement], int]:
    """The layer-based heuristic of Fig. 4; returns (placements, height)."""

    remaining = list(sequence)
    placements: list[Placement] = []
    sky = [0] * strip_width
    line = 0  # current reference line height

    def lowest_space() -> tuple[int, int, int] | None:
        """Lowest run below the reference line: (x, y, width) or None."""

        best = None
        x = 0
        while x < strip_width:
            if sky[x] < line and (best is None or sky[x] < best[1]):
                right = x + 1
                while right < strip_width and sky[right] == sky[x]:
                    right += 1
                best = (x, sky[x], right - x)
            x += 1
        return best

    while remaining:
        # Reference rectangle: first unplaced, long edge along the line.
        ref = remaining.pop(0)
        long_edge = max(ref.width, ref.height)
        short_edge = min(ref.width, ref.height)
        side = long_edge if long_edge <= strip_width else short_edge
        placements.append(
            Placement(
                item_id=ref.item_id,
                x=0,
                y=line,
                width=side,
                height=ref.width * ref.height // side,
                original_width=ref.width,
                original_height=ref.height,
            )
        )
        for x in range(side):
            sky[x] = line + placements[-1].height
        line += placements[-1].height

        # Stack rectangles sharing one side length with the reference edge
        # until the reference line exceeds LB.
        while line < lower_bound:
            match = next(
                (
                    index
                    for index, item in enumerate(remaining)
                    if item.width == side or item.height == side
                ),
                None,
            )
            if match is None:
                break
            item = remaining.pop(match)
            placements.append(
                Placement(
                    item_id=item.item_id,
                    x=0,
                    y=line,
                    width=side,
                    height=item.width * item.height // side,
                    original_width=item.width,
                    original_height=item.height,
                )
            )
            for x in range(side):
                sky[x] = line + placements[-1].height
            line += placements[-1].height

        # Fill the space under the reference line.
        min_side = (
            min(min(item.width, item.height) for item in remaining)
            if remaining
            else 0
        )
        while remaining:
            space = lowest_space()
            if space is None:
                break
            sx, sy, sw = space
            left_top = sky[sx - 1] if sx > 0 else line
            right_top = sky[sx + sw] if sx + sw < strip_width else line
            h1 = left_top - sy
            h2 = right_top - sy
            if sw < min_side or line - sy < min_side:
                # Too small for any unplaced rectangle: raise the space.
                new_y = min(left_top, right_top)
                for x in range(sx, sx + sw):
                    sky[x] = new_y
                continue

            best = None  # (fitness, -index, placement coords)
            for index, item in enumerate(remaining):
                orientations = (
                    (item.width, item.height),
                    (item.height, item.width),
                )
                pair_best = None
                for w, h in orientations:
                    if w > sw or h > line - sy:
                        continue
                    # Fitness (Fig. 3): one point per eliminated corner —
                    # filling the space's width, matching the taller wall's
                    # height, and reaching the reference line.
                    if h1 >= h2:
                        fitness = (w == sw) + (h == h1) + (h == line - sy)
                        px = sx
                    else:
                        fitness = (w == sw) + (h == h2) + (h == line - sy)
                        px = sx + sw - w
                    if pair_best is None or fitness > pair_best[0] or (
                        fitness == pair_best[0]
                        and (w, h) == (item.width, item.height)
                    ):
                        pair_best = (fitness, px, w, h)
                if pair_best is None:
                    continue
                fitness, px, w, h = pair_best
                if best is None or (fitness, -index) > (best[0], best[1]):
                    best = (fitness, -index, index, px, w, h)

            if best is None:
                # Nothing fits this space: raise it.
                new_y = min(left_top, right_top)
                if new_y == sy:
                    # No progress possible; finish the layer.
                    break
                for x in range(sx, sx + sw):
                    sky[x] = new_y
                continue

            _fitness, _neg_index, index, px, w, h = best
            item = remaining.pop(index)
            placements.append(
                Placement(
                    item_id=item.item_id,
                    x=px,
                    y=sy,
                    width=w,
                    height=h,
                    original_width=item.width,
                    original_height=item.height,
                )
            )
            for x in range(px, px + w):
                sky[x] = sy + h
            min_side = (
                min(min(item.width, item.height) for item in remaining)
                if remaining
                else 0
            )

    height = max(placement.top for placement in placements) if placements else 0
    return placements, height


def fast_heuristic(
    items: list[Item], strip_width: int, *, swap_budget: int | None = None
) -> tuple[list[Placement], int]:
    """FH (Fig. 5): perimeter ordering plus a first-improvement swap pass.

    ``swap_budget`` caps the number of swap evaluations (the paper's O(n^2)
    pass is infeasible in plain Python for very large n).
    """

    lower_bound = ceil(
        sum(item.width * item.height for item in items) / strip_width
    )
    ordering = sorted(
        items, key=lambda item: -2 * (item.width + item.height)
    )
    best_placements, best_h = _heuristic_placing(ordering, strip_width, lower_bound)
    best = (best_placements, best_h)

    n = len(ordering)
    budget = n * n if swap_budget is None else swap_budget
    spent = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if spent >= budget:
                return best
            candidate = list(ordering)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            placements, height = _heuristic_placing(
                candidate, strip_width, lower_bound
            )
            spent += 1
            if height < best[1]:
                best = (placements, height)
                ordering = candidate
    return best
