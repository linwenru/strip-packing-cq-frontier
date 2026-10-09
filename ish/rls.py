"""RandomLS: ISH's parameter-free random local search (Algorithm 2).

Four decreasing sorting rules (area, height, width, perimeter) are tried
first and re-ordered by returned height; then, within the time limit, each
rule in that order gets n single-swap hill-climbing iterations (accept when
not worse). The procedure halts early when the naive area lower bound is
reached.
"""

from __future__ import annotations

from math import ceil
from random import Random
from time import perf_counter

from burke_bf.model import Instance, Item, Placement

from .solver import best_fit_pack

SORT_RULES = (
    ("area", lambda item: -item.width * item.height),
    ("height", lambda item: -item.height),
    ("width", lambda item: -item.width),
    ("perimeter", lambda item: -2 * (item.width + item.height)),
)


def random_ls(
    instance: Instance,
    *,
    time_limit: float = 60.0,
    seed: int = 1,
) -> tuple[int | None, list[Placement] | None, float]:
    """Run RandomLS; return (best height, its placements, elapsed seconds)."""

    items = list(instance.items)
    strip_width = instance.strip_width
    rng = Random(seed)
    started = perf_counter()
    n = len(items)
    lower_bound = ceil(sum(item.area for item in items) / strip_width)

    def pack(sequence: list[Item]) -> tuple[list[Placement], int]:
        return best_fit_pack(sequence, strip_width)

    # Initial pass over the four rules; order them by returned height.
    initial: list[tuple[int, str, list[Item], list[Placement]]] = []
    for name, key in SORT_RULES:
        sequence = sorted(items, key=key)
        placements, height = pack(sequence)
        initial.append((height, name, sequence, placements))
    initial.sort(key=lambda entry: (entry[0], entry[1]))

    best_height, best_placements = initial[0][0], initial[0][3]
    if best_height == lower_bound:
        return best_height, best_placements, perf_counter() - started

    while perf_counter() - started < time_limit:
        for _height, _name, sequence, _placements in initial:
            if perf_counter() - started > time_limit:
                break
            current = sequence
            placements, height = pack(current)
            if height < best_height:
                best_height, best_placements = height, placements
            if height == lower_bound:
                return best_height, best_placements, perf_counter() - started
            for _i in range(n):
                if perf_counter() - started > time_limit:
                    break
                a, b = rng.sample(range(n), 2)
                candidate = list(current)
                candidate[a], candidate[b] = candidate[b], candidate[a]
                new_placements, new_height = pack(candidate)
                if new_height <= height:
                    height, current = new_height, candidate
                    if new_height < best_height:
                        best_height, best_placements = new_height, new_placements
                    if new_height == lower_bound:
                        return best_height, best_placements, perf_counter() - started

    return best_height, best_placements, perf_counter() - started
