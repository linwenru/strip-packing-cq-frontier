"""Faithful skyline implementation of Burke et al.'s offline best-fit heuristic."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable

from .model import Instance, Item, Placement, Policy, Solution


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
class _Candidate:
    index: int
    rectangle: _Rectangle
    width: int
    height: int


def _preprocess(items: Iterable[Item]) -> list[_Rectangle]:
    """Rotate to width >= height, then stable-sort as specified in Section 2.2."""

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


def _lowest_gap(skyline: list[int]) -> _Gap:
    """Return the leftmost maximal run at the global minimum skyline height."""

    y = min(skyline)
    x = skyline.index(y)
    right = x + 1
    while right < len(skyline) and skyline[right] == y:
        right += 1
    return _Gap(x=x, y=y, width=right - x)


def _best_fitting(
    rectangles: list[_Rectangle], gap_width: int
) -> _Candidate | None:
    """Find the rectangle dimension that consumes most of a niche.

    The list is ordered by decreasing canonical width. This permits the early
    termination described by Burke et al.: once the canonical width is smaller
    than the best dimension already found, no later rectangle can improve it.
    Equal fits retain the first (therefore larger) rectangle. An exact fit
    terminates immediately.
    """

    best: _Candidate | None = None
    best_fill = 0

    for index, rectangle in enumerate(rectangles):
        if rectangle.width < best_fill:
            break

        if rectangle.width <= gap_width:
            placed_width = rectangle.width
            placed_height = rectangle.height
        elif rectangle.height <= gap_width:
            placed_width = rectangle.height
            placed_height = rectangle.width
        else:
            continue

        candidate = _Candidate(
            index=index,
            rectangle=rectangle,
            width=placed_width,
            height=placed_height,
        )
        if placed_width == gap_width:
            return candidate
        if placed_width > best_fill:
            best = candidate
            best_fill = placed_width

    return best


def _neighbour_heights(
    skyline: list[int], gap: _Gap
) -> tuple[float, float]:
    # The paper treats both strip sides as infinitely tall.
    left = skyline[gap.x - 1] if gap.x > 0 else inf
    right = skyline[gap.right] if gap.right < len(skyline) else inf
    return left, right


def _place_on_left(
    policy: Policy, skyline: list[int], gap: _Gap
) -> bool:
    if policy is Policy.LEFTMOST:
        return True

    left, right = _neighbour_heights(skyline, gap)
    if policy is Policy.TALLEST_NEIGHBOUR:
        return left >= right
    if left == right == inf:
        # Verstichel's clarification of the paper's shortest-neighbour policy:
        # when only the (infinitely tall) sheet sides bound the gap, the
        # rectangle goes to the right-hand side. This also reproduces the
        # paper's Table 3 SN values exactly.
        return False
    return left <= right


def _raise_unfillable_gap(skyline: list[int], gap: _Gap) -> None:
    left, right = _neighbour_heights(skyline, gap)
    new_height = min(left, right)
    if new_height == inf:
        raise RuntimeError("no rectangle fits an empty full-width strip")
    if new_height <= gap.y:
        raise RuntimeError("invalid skyline: neighbour does not bound the gap")
    skyline[gap.x : gap.right] = [int(new_height)] * gap.width


def _placement(
    candidate: _Candidate,
    x: int,
    y: int,
) -> Placement:
    item = candidate.rectangle.item
    return Placement(
        item_id=item.item_id,
        x=x,
        y=y,
        width=candidate.width,
        height=candidate.height,
        original_width=item.width,
        original_height=item.height,
    )


def _fixed_placement(
    old: Placement,
    x: int,
    y: int,
) -> Placement:
    return Placement(
        item_id=old.item_id,
        x=x,
        y=y,
        width=old.height,
        height=old.width,
        original_width=old.original_width,
        original_height=old.original_height,
    )


def _pack(policy: Policy, instance: Instance) -> tuple[list[Placement], list[int]]:
    remaining = _preprocess(instance.items)
    skyline = [0] * instance.strip_width
    placements: list[Placement] = []
    active_gap: _Gap | None = None

    while remaining:
        gap = active_gap if active_gap is not None else _lowest_gap(skyline)
        candidate = _best_fitting(remaining, gap.width)
        if candidate is None:
            _raise_unfillable_gap(skyline, gap)
            active_gap = None
            continue

        on_left = _place_on_left(policy, skyline, gap)
        x = gap.x if on_left else gap.right - candidate.width
        placed = _placement(candidate, x=x, y=gap.y)
        placements.append(placed)
        skyline[x : x + placed.width] = [placed.top] * placed.width
        del remaining[candidate.index]

        remainder = gap.width - placed.width
        if remainder:
            active_gap = _Gap(
                x=gap.x + placed.width if on_left else gap.x,
                y=gap.y,
                width=remainder,
            )
        else:
            active_gap = None

    return placements, skyline


def _highest_placement(placements: list[Placement]) -> Placement:
    # max() is stable, so ties follow the original packing order.
    return max(placements, key=lambda placement: placement.top)


def _reinsert_rotated(
    placement: Placement,
    policy: Policy,
    skyline: list[int],
) -> Placement:
    required_width = placement.height
    while True:
        gap = _lowest_gap(skyline)
        if required_width <= gap.width:
            on_left = _place_on_left(policy, skyline, gap)
            x = gap.x if on_left else gap.right - required_width
            replacement = _fixed_placement(placement, x=x, y=gap.y)
            skyline[x : x + replacement.width] = (
                [replacement.top] * replacement.width
            )
            return replacement
        _raise_unfillable_gap(skyline, gap)


def _postprocess_towers(
    placements: list[Placement],
    skyline: list[int],
    policy: Policy,
) -> tuple[list[Placement], list[int], int]:
    """Rotate and reinsert topmost portrait rectangles while height improves."""

    moves = 0
    while True:
        highest = _highest_placement(placements)
        if highest.width >= highest.height:
            break
        if highest.height > len(skyline):
            # Rotating would exceed the strip width (e.g. BKW03's 8x32 item in
            # a 30-wide strip); the paper gives no rule for this, so stop.
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

        replacement = _reinsert_rotated(highest, policy, skyline)
        placements.append(replacement)
        if max(skyline) >= old_height:
            placements.pop()
            placements.insert(index, highest)
            skyline = old_skyline
            break
        moves += 1

    return placements, skyline, moves


def solve_policy(
    instance: Instance,
    policy: Policy,
    *,
    postprocess: bool = True,
) -> Solution:
    """Pack an instance with one of the paper's three policies."""

    placements, skyline = _pack(policy, instance)
    initial_height = max(skyline)
    tower_moves = 0
    if postprocess:
        placements, skyline, tower_moves = _postprocess_towers(
            placements, skyline, policy
        )

    solution = Solution(
        instance_name=instance.name,
        policy=policy,
        height=max(skyline),
        initial_height=initial_height,
        placements=tuple(placements),
        skyline=tuple(skyline),
        tower_moves=tower_moves,
    )
    validate_solution(instance, solution)
    return solution


def solve(
    instance: Instance,
    *,
    postprocess: bool = True,
) -> tuple[Solution, dict[Policy, Solution]]:
    """Try all three placement policies and return the best solution."""

    by_policy = {
        policy: solve_policy(instance, policy, postprocess=postprocess)
        for policy in Policy
    }
    best = min(by_policy.values(), key=lambda solution: solution.height)
    return best, by_policy


def validate_solution(instance: Instance, solution: Solution) -> None:
    """Raise ``ValueError`` if a solution is incomplete or geometrically invalid."""

    expected = {item.item_id: item for item in instance.items}
    observed = {placement.item_id: placement for placement in solution.placements}
    if observed.keys() != expected.keys():
        missing = sorted(expected.keys() - observed.keys())
        extra = sorted(observed.keys() - expected.keys())
        raise ValueError(f"item mismatch: missing={missing}, extra={extra}")
    if len(observed) != len(solution.placements):
        raise ValueError("an item was placed more than once")

    for placement in solution.placements:
        item = expected[placement.item_id]
        if sorted((placement.width, placement.height)) != sorted(
            (item.width, item.height)
        ):
            raise ValueError(f"item {placement.item_id} has invalid orientation")
        if (
            placement.x < 0
            or placement.y < 0
            or placement.right > instance.strip_width
            or placement.top > solution.height
        ):
            raise ValueError(f"item {placement.item_id} lies outside the strip")

    for index, first in enumerate(solution.placements):
        for second in solution.placements[index + 1 :]:
            separated = (
                first.right <= second.x
                or second.right <= first.x
                or first.top <= second.y
                or second.top <= first.y
            )
            if not separated:
                raise ValueError(
                    f"items {first.item_id} and {second.item_id} overlap"
                )

    geometric_height = max(placement.top for placement in solution.placements)
    if geometric_height != solution.height:
        raise ValueError(
            f"reported height {solution.height} differs from geometric "
            f"height {geometric_height}"
        )
