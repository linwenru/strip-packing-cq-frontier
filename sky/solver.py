"""Wei, Oon, Zhu and Lim's greedy skyline heuristic (EJOR 215:337-346, 2011).

The heuristic packs rectangles one by one into a fixed W x H sheet. A packing
is a rectilinear skyline: a list of horizontal segments with strictly
alternating heights. Rectangles are placed at candidate endpoints of segments
(a left endpoint is a candidate iff the left neighbour is higher, and vice
versa; the outer endpoints are always candidates), possibly bridging over
lower segments. Every step evaluates all feasible position-rectangle pairs
with the paper's priority rules — spread constraint, only fit, minimum local
waste (below / left / right / above), maximum fitness number, earliest in
sequence — and places the best one. After each placement, locally-lowest
segments that no remaining rectangle can stand on are raised and merged.

The rotatable variant evaluates both orientations of each rectangle and
prefers the better one (ties go to the input orientation).

Implementation notes (speed): for each placement step the candidate positions
are prepared once with their maximum feasible width (walk over segments not
higher than the base) and prefix/suffix minima of segment heights, so
feasibility and the resultant spread cost O(1) per rectangle instead of a
full skyline simulation; only the winning placement materializes the updated
skyline.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from time import perf_counter

from burke_bf.model import Item, Placement


@dataclass(frozen=True, slots=True)
class _Segment:
    x: int
    y: int
    length: int

    @property
    def right(self) -> int:
        return self.x + self.length


@dataclass(frozen=True, slots=True)
class _Rect:
    index: int
    item: Item


@dataclass(frozen=True, slots=True)
class _Position:
    """One candidate endpoint with everything needed for O(1) checks."""

    j: int
    side: str  # 'L' = left endpoint of segment j, 'R' = right endpoint
    base_y: int
    max_width: int
    left_wall: float  # height of the left neighbour (sheet_height at the border)
    right_wall: float


def _positions(
    segments: list[_Segment], strip_width: int, sheet_height: int
) -> list[_Position]:
    """Candidate endpoints with precomputed maximum feasible widths."""

    k = len(segments)
    positions: list[_Position] = []
    for j, seg in enumerate(segments):
        left_wall = segments[j - 1].y if j > 0 else sheet_height
        right_wall = segments[j + 1].y if j < k - 1 else sheet_height

        if j == 0 or segments[j - 1].y > seg.y:
            # Left endpoint: bridge rightwards over segments not higher.
            t = j
            while t + 1 < k and segments[t + 1].y <= seg.y:
                t += 1
            max_width = min(strip_width, segments[t].right) - seg.x
            positions.append(
                _Position(j, "L", seg.y, max_width, left_wall, right_wall)
            )
        if j == k - 1 or segments[j + 1].y > seg.y:
            # Right endpoint: bridge leftwards over segments not higher.
            t = j
            while t > 0 and segments[t - 1].y <= seg.y:
                t -= 1
            max_width = seg.right - max(0, segments[t].x)
            positions.append(
                _Position(j, "R", seg.y, max_width, left_wall, right_wall)
            )
    return positions


def _fits_on_segment(rect: _Rect, seg: _Segment, rotatable: bool) -> bool:
    if rotatable:
        return min(rect.item.width, rect.item.height) <= seg.length
    return rect.item.width <= seg.length


def _merge(segments: list[_Segment]) -> list[_Segment]:
    merged: list[_Segment] = []
    for seg in sorted(segments, key=lambda segment: segment.x):
        if merged and merged[-1].right == seg.x and merged[-1].y == seg.y:
            last = merged[-1]
            merged[-1] = _Segment(last.x, last.y, last.length + seg.length)
        else:
            merged.append(seg)
    return merged


def _raise_unfillable(
    segments: list[_Segment], remaining: list[_Rect], rotatable: bool
) -> list[_Segment]:
    """Raise and merge locally-lowest segments no remaining rectangle fits."""

    segments = list(segments)
    while True:
        raised = False
        for j, seg in enumerate(segments):
            left_y = segments[j - 1].y if j > 0 else inf
            right_y = segments[j + 1].y if j < len(segments) - 1 else inf
            if seg.y >= min(left_y, right_y):
                continue
            if any(
                _fits_on_segment(rect, seg, rotatable) for rect in remaining
            ):
                continue
            new_y = min(left_y, right_y)
            if new_y == inf:
                # Nothing can ever stand on the only remaining segment
                # (e.g. unplaceable rectangles); leave it alone.
                continue
            segments = _merge(
                segments[:j]
                + [_Segment(seg.x, int(new_y), seg.length)]
                + segments[j + 1 :]
            )
            raised = True
            break
        if not raised:
            return segments


def _evaluate(
    segments: list[_Segment],
    pos: _Position,
    rect: _Rect,
    w: int,
    h: int,
    sheet_height: int,
    strip_width: int,
    max_spread: float,
    w_min: int,
    h_min: int,
    fit_count: int,
    global_max: int,
    min_upto: list[float],
    min_from: list[float],
) -> tuple | None:
    """Priority key prefix for one oriented placement, or None if infeasible."""

    if w > pos.max_width:
        return None
    base_y = pos.base_y
    if base_y + h > sheet_height:
        return None  # the sheet top is a hard boundary

    seg = segments[pos.j]
    if pos.side == "L":
        rx = seg.x
        covered_from, covered_to = pos.j, pos.j
        while covered_to + 1 < len(segments) and segments[covered_to + 1].x < rx + w:
            covered_to += 1
        resultant_min = min_upto[pos.j]
        if rx + w < segments[covered_to].right:
            resultant_min = min(resultant_min, segments[covered_to].y)
        resultant_min = min(resultant_min, min_from[covered_to + 1], base_y + h)
    else:
        rx = seg.right - w
        covered_from, covered_to = pos.j, pos.j
        while covered_from > 0 and segments[covered_from - 1].right > rx:
            covered_from -= 1
        resultant_min = min_from[pos.j + 1]
        if rx > segments[covered_from].x:
            resultant_min = min(resultant_min, segments[covered_from].y)
        resultant_min = min(resultant_min, min_upto[covered_from], base_y + h)

    if max(global_max, base_y + h) - resultant_min > max_spread:
        return None

    waste = 0
    # (a) waste below: pockets under the rect over covered lower segments.
    for t in range(covered_from, covered_to + 1):
        other = segments[t]
        if other.y < base_y:
            cover = min(other.right, rx + w) - max(other.x, rx)
            waste += (base_y - other.y) * cover

    # (b)/(c) waste beside: leftover of the base segment next to a wall.
    gap_len = seg.length - w
    if 0 < gap_len < w_min:
        wall_y = pos.left_wall if pos.side == "R" else pos.right_wall
        waste += gap_len * (wall_y - base_y)

    # (d) waste above: unusable strip between the rect's top and its wall.
    wall_y = pos.left_wall if pos.side == "L" else pos.right_wall
    gap = wall_y - (base_y + h)
    if 0 < gap < h_min:
        waste += w * gap

    fitness = 0
    if w == seg.length and (base_y != 0 or w == strip_width):
        fitness += 1
    if rx == seg.x and h == pos.left_wall - base_y:
        fitness += 1
    if rx + w == seg.right and h == pos.right_wall - base_y:
        fitness += 1
    if base_y + h == sheet_height:
        fitness += 1

    only_fit = fit_count == 1
    return (0 if only_fit else 1, waste, -fitness)


def skyline_heuristic(
    sequence: list[Item],
    strip_width: int,
    sheet_height: int,
    max_spread: float,
    rotatable: bool = True,
    deadline: float | None = None,
) -> tuple[list[Placement], float, bool]:
    """Run the greedy heuristic; return (placements, area utilization, success)."""

    remaining = [_Rect(index, item) for index, item in enumerate(sequence)]
    segments: list[_Segment] = [_Segment(0, 0, strip_width)]
    placements: list[Placement] = []
    packed_area = 0

    while remaining:
        if deadline is not None and perf_counter() > deadline:
            break
        positions = _positions(segments, strip_width, sheet_height)
        global_max = max(seg.y for seg in segments)
        k = len(segments)
        min_upto = [inf] * (k + 1)  # min segment y strictly before index t
        for t in range(k):
            min_upto[t + 1] = min(min_upto[t], segments[t].y)
        min_from = [inf] * (k + 1)  # min segment y from index t onwards
        for t in range(k - 1, -1, -1):
            min_from[t] = min(min_from[t + 1], segments[t].y)

        # Count fitting rectangles per position for the only-fit rule.
        fit_counts: dict[tuple[int, str], tuple[int, _Rect | None]] = {}
        for pos in positions:
            count = 0
            single: _Rect | None = None
            for rect in remaining:
                if rotatable:
                    fits = (
                        rect.item.width <= pos.max_width
                        or rect.item.height <= pos.max_width
                    )
                else:
                    fits = rect.item.width <= pos.max_width
                if fits:
                    count += 1
                    single = rect
            fit_counts[(pos.j, pos.side)] = (count, single)

        if rotatable:
            w_min = min(
                min(rect.item.width, rect.item.height) for rect in remaining
            )
            h_min = w_min
        else:
            w_min = min(rect.item.width for rect in remaining)
            h_min = min(rect.item.height for rect in remaining)

        best = None  # (key, rect, w, h, pos, rx)
        # Only-fit placements outrank everything (rule 2), but only when they
        # are actually feasible; if every only-fit candidate is rejected by
        # the spread/height constraints, the full space must be evaluated.
        only_space = [
            (pos, fit_counts[(pos.j, pos.side)][1])
            for pos in positions
            if fit_counts[(pos.j, pos.side)][0] == 1
        ]
        eval_spaces = [only_space] if only_space else []
        eval_spaces.append([(pos, rect) for pos in positions for rect in remaining])
        for eval_space in eval_spaces:
            for pos, rect in eval_space:
                if rect is None:
                    continue
                fit_count, fit_single = fit_counts[(pos.j, pos.side)]
                orientations = (
                    ((rect.item.width, rect.item.height), (rect.item.height, rect.item.width))
                    if rotatable
                    else ((rect.item.width, rect.item.height),)
                )
                pair_best = None
                for w, h in orientations:
                    key = _evaluate(
                        segments, pos, rect, w, h, sheet_height,
                        strip_width, max_spread, w_min, h_min,
                        fit_count if fit_single is rect else 2, global_max,
                        min_upto, min_from,
                    )
                    if key is None:
                        continue
                    if pair_best is None or key < pair_best[0] or (
                        key == pair_best[0]
                        and (w, h) == (rect.item.width, rect.item.height)
                    ):
                        pair_best = (key, w, h)
                if pair_best is None:
                    continue
                key, w, h = pair_best
                rx = segments[pos.j].x if pos.side == "L" else segments[pos.j].right - w
                full_key = key + (rect.index, pos.base_y, rx)
                if best is None or full_key < best[0]:
                    best = (full_key, rect, w, h, pos, rx)
            if best is not None:
                break

        if best is None:
            break  # failure: nothing can be placed anywhere

        _key, rect, w, h, pos, rx = best
        base_y = pos.base_y
        placements.append(
            Placement(
                item_id=rect.item.item_id,
                x=rx,
                y=base_y,
                width=w,
                height=h,
                original_width=rect.item.width,
                original_height=rect.item.height,
            )
        )
        packed_area += rect.item.area
        remaining = [other for other in remaining if other is not rect]

        # Materialize the updated skyline (step 1) for the winning placement.
        new_segments: list[_Segment] = []
        for other in segments:
            if other.right <= rx or other.x >= rx + w:
                new_segments.append(other)
                continue
            if other.x < rx:
                new_segments.append(_Segment(other.x, other.y, rx - other.x))
            if other.right > rx + w:
                new_segments.append(
                    _Segment(rx + w, other.y, other.right - (rx + w))
                )
        new_segments.append(_Segment(rx, base_y + h, w))
        segments = _raise_unfillable(
            _merge(new_segments), remaining, rotatable
        )

    return placements, packed_area / (strip_width * sheet_height), not remaining


def skyline_array(
    placements: list[Placement], strip_width: int
) -> tuple[int, ...]:
    heights = [0] * strip_width
    for placement in placements:
        for x in range(placement.x, placement.right):
            heights[x] = max(heights[x], placement.top)
    return tuple(heights)
