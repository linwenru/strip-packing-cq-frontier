"""Partial-state continuation for the CCH engine (OFF-2 validations).

The engine itself (``cch/solver.py``) stays at the OFF-1 reproduction state;
OFF-2 additions live in separate modules so OFF-1 reruns and the public
mirror are untouched. The continuation loop here is rebuilt on the engine's
own primitives (``_lowest_gap``/``_select``/``_place_on_left``/``_raise``/
``_reduce_towers``); ``python3 -m cch.off2_validation accept-state`` verifies
on every intermediate state of the 36 development instances that a truncated
run resumed from a snapshot reproduces the one-shot ``solve_config`` run
exactly (post-checkpoint trajectory, orientations, final layout, validity).

Unsupported (``ValueError``): ``vertical_niche`` (the niche pipeline is
engine-internal and OFF-2 continuations never use it) and ``rotation_rule``
(its 2n configuration list is not representable as plain items).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from burke_bf.model import Instance, Item, Placement

from .model import Config, Selection
from .solver import (
    _lowest_gap,
    _neighbours,
    _place_on_left,
    _raise,
    _Rectangle,
    _reduce_towers,
    _select,
    _sort,
)


@dataclass(frozen=True, slots=True)
class EngineState:
    """Mid-run snapshot after a placement: skyline, placements, remaining items.

    ``remaining`` keeps the engine's list order as plain items (before
    orientation normalisation); feeding the three fields to
    ``solve_config_from_state`` reproduces the continuation.
    """

    skyline: tuple[int, ...]
    placements: tuple[Placement, ...]
    remaining: tuple[Item, ...]


def _check_supported(config: Config) -> None:
    if config.vertical_niche:
        raise ValueError("vertical_niche states cannot be resumed/traced here")
    if config.rotation_rule:
        raise ValueError("rotation_rule states cannot be resumed/traced here")


def _select_any_order(
    config: Config, remaining: list[_Rectangle], skyline: list[int], gap
):
    """``solver._select`` without the width-sorted-only early break.

    The engine's widest-fit scan breaks at the first rectangle narrower than
    the current best fill, which is sound only on width-sorted lists (fresh
    engine runs). Continuations and sequence-driven walks may hold arbitrarily
    ordered lists, where the break silently degrades widest-fit to a
    prefix-local scan (found by code inspection, 2026-09-17). The widest-fit
    branch is replicated here as a full scan (identical results on sorted
    lists); other selection rules have no order-sensitive break and delegate.
    """

    if config.selection is not Selection.WIDEST_FIT:
        return _select(config, remaining, skyline, gap)
    best = None
    best_fill = 0
    for index, r in enumerate(remaining):
        if r.width <= gap.width:
            w, h = r.width, r.height
        elif config.rotatable and r.height <= gap.width:
            w, h = r.height, r.width
        else:
            continue
        if w == gap.width:
            return index, r, w, h
        if w > best_fill:
            best = (index, r, w, h)
            best_fill = w
    return best


def _normalise(items: list[Item] | tuple[Item, ...], rotatable: bool) -> list[_Rectangle]:
    if rotatable:
        return [
            _Rectangle(item, max(item.width, item.height), min(item.width, item.height))
            for item in items
        ]
    return [_Rectangle(item, item.width, item.height) for item in items]


def _pack_loop(
    skyline: list[int],
    placements: list[Placement],
    remaining: list[_Rectangle],
    config: Config,
    trace: list[EngineState] | None = None,
) -> None:
    """Lowest-gap loop mirroring the engine's plain path (no niche pipeline)."""

    while remaining:
        gap = _lowest_gap(skyline)
        found = _select_any_order(config, remaining, skyline, gap)
        if found is None:
            _raise(skyline, gap)
            continue
        index, rectangle, width, height = found
        y = gap.y
        if config.selection is Selection.FITNESS_NUMBER:
            # ISH's corner rule: the corner with the better fitness, ties
            # to the tallest neighbour (then to the left corner).
            left, right = _neighbours(skyline, gap)
            lh = left - gap.y if left != inf else inf
            rh = right - gap.y if right != inf else inf
            fit_left = (width == gap.width) + (height == lh) + (
                width == gap.width and height == rh
            )
            fit_right = (width == gap.width) + (height == rh) + (
                width == gap.width and height == lh
            )
            if fit_left != fit_right:
                on_left = fit_left > fit_right
            else:
                on_left = lh >= rh
        else:
            on_left = _place_on_left(config.placement, skyline, gap, gap.y + height)
        x = gap.x if on_left else gap.right - width
        placements.append(
            Placement(
                item_id=rectangle.item.item_id,
                x=x,
                y=y,
                width=width,
                height=height,
                original_width=rectangle.item.width,
                original_height=rectangle.item.height,
            )
        )
        skyline[x : x + width] = [y + height] * width
        del remaining[index]
        if trace is not None:
            trace.append(
                EngineState(
                    skyline=tuple(skyline),
                    placements=tuple(placements),
                    remaining=tuple(r.item for r in remaining),
                )
            )


def _pack_and_reduce(
    skyline: list[int],
    placements: list[Placement],
    remaining: list[_Rectangle],
    config: Config,
    trace: list[EngineState] | None = None,
) -> tuple[list[Placement], list[int], int]:
    _pack_loop(skyline, placements, remaining, config, trace)
    moves = 0
    if config.tower_removal:
        placements, skyline, moves = _reduce_towers(
            placements, skyline, config.placement
        )
    return placements, skyline, moves


def solve_config_traced(
    instance: Instance,
    config: Config,
    sequence: list[Item] | None = None,
) -> tuple[list[Placement], list[int], int, list[EngineState]]:
    """Fresh run returning ``(placements, skyline, moves, trace)``.

    ``trace[k]`` is the engine state after k+1 placements. ``sequence`` keeps
    the engine's semantics: items are packed in the given order, only
    orientation normalisation applies, the configured ordering is bypassed.
    """

    _check_supported(config)
    if sequence is not None:
        remaining = _normalise(sequence, config.rotatable)
    else:
        remaining = _sort(
            list(instance.items), config.ordering, instance.strip_width,
            config.rotatable,
        )
    skyline = [0] * instance.strip_width
    placements: list[Placement] = []
    trace: list[EngineState] = []
    placements, skyline, moves = _pack_and_reduce(
        skyline, placements, remaining, config, trace
    )
    return placements, skyline, moves, trace


def solve_config_from_state(
    skyline: tuple[int, ...] | list[int],
    placed: tuple[Placement, ...] | list[Placement],
    remaining: tuple[Item, ...] | list[Item],
    config: Config,
) -> tuple[list[Placement], list[int], int]:
    """Continue packing from a partial state (checkpoint continuation).

    ``remaining`` items are packed in the given list order; only orientation
    normalisation applies (the configured ordering is bypassed), matching the
    ``sequence`` semantics of ``solve_config``. All inputs are copied up
    front, so independent continuations from the same state never share
    mutable state. Returns ``(placements, skyline, moves)`` with
    ``placements`` including the prefix and ``moves`` counting the
    continuation's tower moves only.
    """

    _check_supported(config)
    cont_skyline = [int(value) for value in skyline]
    placements = list(placed)
    rectangles = _normalise(list(remaining), config.rotatable)
    return _pack_and_reduce(cont_skyline, placements, rectangles, config)
