"""Verstichel, De Causmaecker and Vanden Berghe's three-way best-fit heuristic.

The heuristic (ITOR 20(5):711-730, 2013; reproduced from the open-access
summary in Appendix A of Verstichel's PhD thesis) extends Burke's best-fit
in three ways: two extra input orderings (decreasing height and surface
alongside width), three extra placement policies (rightmost, MinDiff and
MaxDiff alongside leftmost, tallest- and shortest-neighbour), and a rotation
rule that substitutes the rotated configuration for oversized rectangles
(wider than the strip), so they enter the packing early. Every one of the 18
ordering x policy combinations is solved (with tower reduction) and the best
solution is returned.

The rectangle list holds both configurations of each rectangle; the selected
rectangle is the first in the list whose horizontal dimension fits the gap,
placed with its listed orientation, and both its configurations are then
removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from burke_bf.model import Instance, Item, Placement
from burke_bf.solver import (
    _lowest_gap,
    _neighbour_heights,
    _raise_unfillable_gap,
    validate_solution,
)

from .model import (
    Combination,
    Ordering,
    Solution,
    TwPolicy,
    all_combinations,
)


@dataclass(frozen=True, slots=True)
class _Entry:
    """One configuration of a rectangle in the ordered list."""

    item: Item
    horiz: int
    vert: int


def _preprocess(
    items: list[Item] | tuple[Item, ...], ordering: Ordering, strip_width: int
) -> list[_Entry]:
    """Build the ordered configuration list for one ordering.

    Rectangles are canonicalised (width >= height). Oversized rectangles
    (canonical width larger than the strip) follow the paper's rotation rule,
    second approach: the default configuration is replaced by the rotated
    one, so they appear early in the list and are placed early.
    """

    canonical = [
        (item, max(item.width, item.height), min(item.width, item.height))
        for item in items
    ]
    entries: list[_Entry] = []
    if ordering is Ordering.WIDTH:
        for item, width, height in canonical:
            entries.append(_Entry(item, width, height))
            entries.append(_Entry(item, height, width))
        entries.sort(key=lambda entry: (-entry.horiz, -entry.vert))
        # Rotation rule, second approach: substitute the rotated configuration
        # for the oversized default one in place, so it is placed early. Only
        # the default configuration has horiz > strip width.
        return [
            _Entry(entry.item, entry.vert, entry.horiz)
            if entry.horiz > strip_width
            else entry
            for entry in entries
        ]

    if ordering is Ordering.HEIGHT:
        # Ties by decreasing width, the BF-consistent secondary key; variant
        # tests against Table A.3 confirm it over plain stable order.
        canonical.sort(key=lambda triple: (-triple[2], -triple[1]))
    else:
        canonical.sort(key=lambda triple: (-triple[1] * triple[2], -triple[1]))
    for item, width, height in canonical:
        if width > strip_width:
            entries.append(_Entry(item, height, width))
        else:
            entries.append(_Entry(item, width, height))
        entries.append(_Entry(item, height, width))
    return entries


def _place_on_left(
    policy: TwPolicy, skyline: list[int], gap, placed_top: int
) -> bool:
    """Decide whether the rectangle goes at the left-hand side of the gap."""

    if policy is TwPolicy.LEFTMOST:
        return True
    if policy is TwPolicy.RIGHTMOST:
        return False

    left, right = _neighbour_heights(skyline, gap)
    if policy is TwPolicy.TALLEST:
        return left >= right
    if policy is TwPolicy.MAX_DIFF:
        return abs(left - placed_top) >= abs(right - placed_top)

    # SHORTEST and MIN_DIFF never treat the infinitely tall sheet side as
    # the shortest neighbour; bounded on both sides by sheet sides, the
    # rectangle goes to the right (the paper's clarification).
    if left == right == inf:
        return False
    if policy is TwPolicy.SHORTEST:
        return left <= right
    return abs(left - placed_top) <= abs(right - placed_top)


def _pack(
    combination: Combination, instance: Instance
) -> tuple[list[Placement], list[int]]:
    remaining = _preprocess(instance.items, combination.ordering, instance.strip_width)
    skyline = [0] * instance.strip_width
    placements: list[Placement] = []

    while remaining:
        gap = _lowest_gap(skyline)
        chosen = next(
            (index for index, entry in enumerate(remaining)
             if entry.horiz <= gap.width),
            None,
        )
        if chosen is None:
            _raise_unfillable_gap(skyline, gap)
            continue

        entry = remaining[chosen]
        top = gap.y + entry.vert
        on_left = _place_on_left(combination.policy, skyline, gap, top)
        x = gap.x if on_left else gap.right - entry.horiz
        placements.append(
            Placement(
                item_id=entry.item.item_id,
                x=x,
                y=gap.y,
                width=entry.horiz,
                height=entry.vert,
                original_width=entry.item.width,
                original_height=entry.item.height,
            )
        )
        skyline[x : x + entry.horiz] = [top] * entry.horiz
        remaining = [
            candidate
            for candidate in remaining
            if candidate.item.item_id != entry.item.item_id
        ]

    return placements, skyline


def _reduce_towers(
    placements: list[Placement],
    skyline: list[int],
    policy: TwPolicy,
) -> tuple[list[Placement], list[int], int]:
    """Burke's tower removal: rotate and reinsert topmost portrait rectangles."""

    moves = 0
    while True:
        highest = max(placements, key=lambda placement: placement.top)
        if highest.width >= highest.height:
            break
        if highest.height > len(skyline):
            # Rotating would exceed the strip width; cannot improve.
            break

        old_height = max(skyline)
        old_skyline = skyline.copy()
        index = placements.index(highest)
        if any(
            value != highest.top
            for value in skyline[highest.x : highest.right]
        ):
            raise RuntimeError(
                "highest rectangle is not exposed across its full width"
            )
        skyline[highest.x : highest.right] = [
            value - highest.height
            for value in skyline[highest.x : highest.right]
        ]
        del placements[index]

        required_width = highest.height
        while True:
            gap = _lowest_gap(skyline)
            if required_width <= gap.width:
                top = gap.y + highest.width
                on_left = _place_on_left(policy, skyline, gap, top)
                x = gap.x if on_left else gap.right - required_width
                replacement = Placement(
                    item_id=highest.item_id,
                    x=x,
                    y=gap.y,
                    width=highest.height,
                    height=highest.width,
                    original_width=highest.original_width,
                    original_height=highest.original_height,
                )
                skyline[x : x + replacement.width] = [top] * replacement.width
                break
            _raise_unfillable_gap(skyline, gap)

        placements.append(replacement)
        if max(skyline) >= old_height:
            placements.pop()
            placements.insert(index, highest)
            skyline = old_skyline
            break
        moves += 1

    return placements, skyline, moves


def solve_combination(
    instance: Instance, combination: Combination, *, validate: bool = True
) -> Solution:
    """Pack an instance under one ordering x policy combination."""

    placements, skyline = _pack(combination, instance)
    initial_height = max(skyline)
    placements, skyline, tower_moves = _reduce_towers(
        placements, skyline, combination.policy
    )
    solution = Solution(
        instance_name=instance.name,
        combination=combination,
        height=max(skyline),
        initial_height=initial_height,
        placements=tuple(placements),
        skyline=tuple(skyline),
        tower_moves=tower_moves,
    )
    if validate:
        validate_solution(instance, solution)
    return solution


def solve(
    instance: Instance, *, validate_each: bool = False
) -> tuple[Solution, dict[Combination, int]]:
    """Test all 18 combinations and return the best solution + heights.

    Only the best solution is geometry-checked by default; validating every
    combination costs O(n^2) apiece, which is prohibitive for large instances.
    """

    best: Solution | None = None
    heights: dict[Combination, int] = {}
    for combination in all_combinations():
        solution = solve_combination(instance, combination, validate=validate_each)
        heights[combination] = solution.height
        if best is None or solution.height < best.height:
            best = solution
    assert best is not None  # all_combinations is never empty
    validate_solution(instance, best)
    return best, heights


def solve(
    instance: Instance, *, validate_each: bool = False
) -> tuple[Solution, dict[Combination, int]]:
    """Test all 18 combinations and return the best solution + heights.

    Only the best solution is geometry-checked by default; validating every
    combination costs O(n^2) apiece, which is prohibitive for large instances.
    """

    best: Solution | None = None
    heights: dict[Combination, int] = {}
    for combination in all_combinations():
        solution = solve_combination(instance, combination)
        heights[combination] = solution.height
        if best is None or solution.height < best.height:
            best = solution
    assert best is not None  # all_combinations is never empty
    validate_solution(instance, best)
    return best, heights
