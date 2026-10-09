"""Faithful implementation of Aşık and Özcan's bidirectional best-fit heuristic.

The semantics follow the paper (Annals of Operations Research 172:405-427,
2009): an expected best height (EBH) is computed up front, and every policy
combination is tested exhaustively. Each iteration considers the lowest
horizontal gap and a single vertical niche — the region above the leftmost
skyline segment still below the EBH — through an exact-fit stage, a best-fit
stage, and a gap-raising stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf
from typing import Iterable

from burke_bf.model import Instance, Item, Placement, Policy
from burke_bf.solver import validate_solution

from .model import (
    BestFitOrder,
    Combination,
    ExactOrder,
    HorizontalBestFit,
    HorizontalExact,
    Solution,
    VerticalBestFit,
    VerticalExact,
    all_combinations,
)


@dataclass(frozen=True, slots=True)
class _Rectangle:
    item: Item
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _Gap:
    x: int
    y: int
    width: int

    @property
    def right(self) -> int:
        return self.x + self.width


@dataclass(frozen=True, slots=True)
class _VerticalNiche:
    """Region above one skyline segment, extending up to the expected height."""

    x: int
    y: int
    width: int  # horizontal extent (the segment's width)
    vheight: int  # vertical extent, up to the expected best height


@dataclass(frozen=True, slots=True)
class _Candidate:
    index: int
    rectangle: _Rectangle
    x: int
    y: int
    width: int
    height: int


def _preprocess(items: Iterable[Item]) -> list[_Rectangle]:
    """Rotate to width >= height, then stable-sort as in the BF paper."""

    rectangles = [
        _Rectangle(
            item=item,
            width=max(item.width, item.height),
            height=min(item.width, item.height),
        )
        for item in items
    ]
    rectangles.sort(key=lambda rectangle: (-rectangle.width, -rectangle.height))
    return rectangles


def _expected_best_height(instance: Instance) -> int:
    """Continuous bound, lifted to the maximum side length when needed."""

    bound = ceil(instance.total_area / instance.strip_width)
    max_side = max(
        max(item.width, item.height) for item in instance.items
    )
    if max_side > bound and max_side > instance.strip_width:
        return max_side
    return bound


def _lowest_gap(skyline: list[int]) -> _Gap:
    """Return the leftmost maximal run at the global minimum skyline height."""

    y = min(skyline)
    x = skyline.index(y)
    right = x + 1
    while right < len(skyline) and skyline[right] == y:
        right += 1
    return _Gap(x=x, y=y, width=right - x)


def _vertical_niche(skyline: list[int], ebh: int) -> _VerticalNiche | None:
    """Region above the leftmost segment not yet reaching the expected height.

    Derived from Figure 5 and the M1 walkthrough (Table 2): scan the skyline
    segments left to right; the first segment below the EBH hosts the niche,
    which spans the segment horizontally and reaches the EBH vertically.
    """

    x = 0
    while x < len(skyline):
        right = x + 1
        while right < len(skyline) and skyline[right] == skyline[x]:
            right += 1
        if skyline[x] < ebh:
            return _VerticalNiche(
                x=x, y=skyline[x], width=right - x, vheight=ebh - skyline[x]
            )
        x = right
    return None


def _neighbour_heights(skyline: list[int], gap: _Gap) -> tuple[float, float]:
    # The paper treats both strip sides as infinitely tall.
    left = skyline[gap.x - 1] if gap.x > 0 else inf
    right = skyline[gap.right] if gap.right < len(skyline) else inf
    return left, right


def _raise_unfillable_gap(skyline: list[int], gap: _Gap) -> None:
    left, right = _neighbour_heights(skyline, gap)
    new_height = min(left, right)
    if new_height == inf:
        raise RuntimeError("no rectangle fits an empty full-width strip")
    if new_height <= gap.y:
        raise RuntimeError("invalid skyline: neighbour does not bound the gap")
    skyline[gap.x : gap.right] = [int(new_height)] * gap.width


def _place_on_left(policy: Policy, skyline: list[int], gap: _Gap) -> bool:
    if policy is Policy.LEFTMOST:
        return True

    left, right = _neighbour_heights(skyline, gap)
    if policy is Policy.TALLEST_NEIGHBOUR:
        return left >= right
    # SN: Burke's text is ambiguous when both neighbours are the strip
    # sides. We tested both readings against the paper's Table 6 (35
    # instances, 2026-08 re-examination): tie-left matches 20 exactly,
    # Verstichel's tie-right clarification only 17 (BKW08/C4P3/C7P2 move
    # by ±1). The paper's own text is silent on the tie; the left reading
    # is kept as the one with better agreement.
    return left <= right


def _tre(
    rectangles: list[_Rectangle], gap: _Gap
) -> tuple[int, _Rectangle, int, int] | None:
    """First rectangle in list order that fills the gap width exactly."""

    for index, rectangle in enumerate(rectangles):
        if rectangle.width == gap.width:
            return index, rectangle, rectangle.width, rectangle.height
        if rectangle.height == gap.width:
            return index, rectangle, rectangle.height, rectangle.width
    return None


def _exact_horizontal(
    rectangles: list[_Rectangle],
    skyline: list[int],
    gap: _Gap,
    policy: HorizontalExact,
    placement: Policy,
) -> _Candidate | None:
    if policy is HorizontalExact.TRE:
        found = _tre(rectangles, gap)
        if found is None:
            return None
        index, rectangle, width, height = found
        return _Candidate(index, rectangle, gap.x, gap.y, width, height)

    # NRE: try a rectangle whose height levels the tallest neighbour's top,
    # then the shortest one's, so the plateau widens (Figure 7); otherwise
    # fall back to TRE. Placements into the gap use placement policy#7, which
    # Table 1 defines as the policy for the horizontal gap; variant tests
    # against the paper's Table 6 confirm this reading.
    left, right = _neighbour_heights(skyline, gap)
    attempts = (
        (max(left, right), left >= right),  # tallest neighbour
        (min(left, right), left <= right),  # shortest neighbour
    )
    for neighbour_height, _on_left in attempts:
        if neighbour_height == inf:
            continue
        target = neighbour_height - gap.y
        for index, rectangle in enumerate(rectangles):
            if rectangle.height == target and rectangle.width <= gap.width:
                width, height = rectangle.width, rectangle.height
            elif rectangle.width == target and rectangle.height <= gap.width:
                width, height = rectangle.height, rectangle.width
            else:
                continue
            on_left = _place_on_left(placement, skyline, gap)
            x = gap.x if on_left else gap.right - width
            return _Candidate(index, rectangle, x, gap.y, width, height)

    found = _tre(rectangles, gap)
    if found is None:
        return None
    index, rectangle, width, height = found
    return _Candidate(index, rectangle, gap.x, gap.y, width, height)


def _exact_vertical(
    rectangles: list[_Rectangle], niche: _VerticalNiche
) -> _Candidate | None:
    """Exact fit into the vertical niche, placed transposed at its bottom-left.

    A rectangle is placed only when its width equals the niche's vertical
    extent, so it spans the niche up to the EBH (and its height fits the
    niche's width). The paper's fallback sentence ("height as close as
    possible to the height of the vertical niche") is ambiguous, but every
    non-exact reading contradicts the M1 walkthrough in Table 2, so the
    exact-only reading is implemented.
    """

    for index, rectangle in enumerate(rectangles):
        if (
            rectangle.width == niche.vheight
            and rectangle.height <= niche.width
        ):
            return _Candidate(
                index, rectangle, niche.x, niche.y,
                rectangle.height, rectangle.width,
            )
    return None


def _best_horizontal(
    rectangles: list[_Rectangle],
    skyline: list[int],
    gap: _Gap,
    policy: HorizontalBestFit,
) -> _Candidate | None:
    if policy is HorizontalBestFit.FP:
        # First fitting rectangle in list order; often a tall rotated one.
        for index, rectangle in enumerate(rectangles):
            if rectangle.width <= gap.width:
                width, height = rectangle.width, rectangle.height
            elif rectangle.height <= gap.width:
                width, height = rectangle.height, rectangle.width
            else:
                continue
            return _Candidate(index, rectangle, 0, 0, width, height)
        return None

    # BP: Burke's best fit — minimise the remaining gap, exact fit wins early.
    best: _Candidate | None = None
    best_fill = 0
    for index, rectangle in enumerate(rectangles):
        if rectangle.width < best_fill:
            break
        if rectangle.width <= gap.width:
            width, height = rectangle.width, rectangle.height
        elif rectangle.height <= gap.width:
            width, height = rectangle.height, rectangle.width
        else:
            continue
        if width == gap.width:
            return _Candidate(index, rectangle, 0, 0, width, height)
        if width > best_fill:
            best = _Candidate(index, rectangle, 0, 0, width, height)
            best_fill = width
    return best


def _best_vertical(
    rectangles: list[_Rectangle],
    niche: _VerticalNiche,
    policy: VerticalBestFit,
) -> _Candidate | None:
    for index, rectangle in enumerate(rectangles):
        if rectangle.width > niche.vheight:
            continue
        if policy is VerticalBestFit.FH:
            # Height fixed to the niche width; the sorted list then yields
            # the largest width.
            if rectangle.height != niche.width:
                continue
        elif rectangle.height > niche.width:
            # WR: widest rectangle that fits, height within the niche width.
            continue
        return _Candidate(
            index, rectangle, niche.x, niche.y,
            rectangle.height, rectangle.width,
        )
    return None


def _pack(
    combination: Combination, instance: Instance, ebh: int
) -> tuple[list[Placement], list[int]]:
    remaining = _preprocess(instance.items)
    skyline = [0] * instance.strip_width
    placements: list[Placement] = []

    while remaining:
        gap = _lowest_gap(skyline)
        niche = _vertical_niche(skyline, ebh)
        candidate = _search_stages(combination, remaining, skyline, gap, niche)
        if candidate is None:
            _raise_unfillable_gap(skyline, gap)
            continue

        rectangle = candidate.rectangle
        placements.append(
            Placement(
                item_id=rectangle.item.item_id,
                x=candidate.x,
                y=candidate.y,
                width=candidate.width,
                height=candidate.height,
                original_width=rectangle.item.width,
                original_height=rectangle.item.height,
            )
        )
        skyline[candidate.x : candidate.x + candidate.width] = [
            candidate.y + candidate.height
        ] * candidate.width
        del remaining[candidate.index]

    return placements, skyline


def _search_stages(
    combination: Combination,
    remaining: list[_Rectangle],
    skyline: list[int],
    gap: _Gap,
    niche: _VerticalNiche | None,
) -> _Candidate | None:
    """Exact-fit stage, then best-fit stage, honouring the ordering policies."""

    exact_horizontal_first = combination.exact_order is ExactOrder.EHV
    for horizontal_first in (exact_horizontal_first, not exact_horizontal_first):
        if horizontal_first:
            candidate = _exact_horizontal(
                remaining, skyline, gap, combination.horizontal_exact,
                combination.placement,
            )
        else:
            candidate = None
            if (
                combination.vertical_exact is VerticalExact.ENABLED
                and niche is not None
            ):
                candidate = _exact_vertical(remaining, niche)
        if candidate is not None:
            return candidate

    best_horizontal_first = combination.best_order is BestFitOrder.BHV
    for horizontal_first in (best_horizontal_first, not best_horizontal_first):
        if horizontal_first:
            candidate = _best_horizontal(
                remaining, skyline, gap, combination.horizontal_best
            )
            if candidate is not None:
                on_left = _place_on_left(combination.placement, skyline, gap)
                x = gap.x if on_left else gap.right - candidate.width
                candidate = _Candidate(
                    candidate.index, candidate.rectangle, x, gap.y,
                    candidate.width, candidate.height,
                )
        else:
            candidate = None
            if (
                combination.vertical_best is not VerticalBestFit.NO_VB
                and niche is not None
            ):
                candidate = _best_vertical(
                    remaining, niche, combination.vertical_best
                )
        if candidate is not None:
            return candidate

    return None


def solve_combination(
    instance: Instance, combination: Combination, *, validate: bool = True
) -> Solution:
    """Pack an instance under one policy combination."""

    placements, skyline = _pack(combination, instance, _expected_best_height(instance))
    solution = Solution(
        instance_name=instance.name,
        combination=combination,
        height=max(skyline),
        placements=tuple(placements),
        skyline=tuple(skyline),
    )
    if validate:
        validate_solution(instance, solution)
    return solution


def solve(
    instance: Instance, *, validate_each: bool = False
) -> tuple[Solution, dict[Combination, int]]:
    """Test all policy combinations and return the best solution + heights.

    Only the best solution is geometry-checked by default; validating every
    combination costs O(n^2) apiece, which is prohibitive for large instances.
    """

    best: Solution | None = None
    heights: dict[Combination, int] = {}
    for combination in all_combinations():
        solution = solve_combination(
            instance, combination, validate=validate_each
        )
        heights[combination] = solution.height
        if best is None or solution.height < best.height:
            best = solution
    assert best is not None  # all_combinations is never empty
    validate_solution(instance, best)
    return best, heights
