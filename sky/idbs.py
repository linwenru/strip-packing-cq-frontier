"""2DRPSolver (tabu search) and IDBS (iterative doubling binary search).

Algorithm 1 of Wei et al. (2011): for each of six sorting rules and four
max_spread values, run the greedy heuristic; on failure, improve the input
sequence by tabu search (10 swap neighbours per iteration, tabu tenure 3n).
Algorithm 2: binary search on the sheet height H between the naive lower
bound and LB x 1.1, doubling the tabu iteration budget after each attempt.
"""

from __future__ import annotations

from math import ceil, sqrt
from random import Random
from time import perf_counter

from burke_bf.model import Instance, Item, Placement

from .solver import skyline_heuristic

SORT_RULES = (
    ("area", lambda item: -item.width * item.height),
    ("width", lambda item: -item.width),
    ("height", lambda item: -item.height),
    ("perimeter", lambda item: -2 * (item.width + item.height)),
    ("maxside", lambda item: -max(item.width, item.height)),
    (
        "diagonal",
        lambda item: -(sqrt(item.width**2 + item.height**2) + item.width + item.height),
    ),
)


def _spread_values(items: list[Item], sheet_height: int) -> list[float]:
    mh = max(item.height for item in items)
    return (
        float(mh),
        mh + (sheet_height - mh) / 3,
        mh + 2 * (sheet_height - mh) / 3,
        float(sheet_height),
    )


def solver_2drp(
    items: list[Item],
    strip_width: int,
    sheet_height: int,
    iter_max: int,
    rng: Random,
    rotatable: bool = True,
    use_tabu: bool = True,
    deadline: float | None = None,
) -> tuple[bool, list[Placement] | None, float]:
    """Run Algorithm 1; return (success, best placements, best utilization)."""

    best_placements: list[Placement] | None = None
    best_util = -1.0

    def record(result: tuple[list[Placement], float, bool]) -> bool:
        nonlocal best_placements, best_util
        placements, util, success = result
        if util > best_util:
            best_placements, best_util = placements, util
        return success

    n = len(items)
    for _name, key in SORT_RULES:
        if deadline is not None and perf_counter() > deadline:
            break
        sequence = sorted(items, key=key)
        for max_spread in _spread_values(items, sheet_height):
            if deadline is not None and perf_counter() > deadline:
                break
            if record(skyline_heuristic(
                sequence, strip_width, sheet_height, max_spread, rotatable,
                deadline=deadline,
            )):
                return True, best_placements, best_util
            if not use_tabu:
                continue
            current = sequence
            tabu: dict[tuple[str, str], int] = {}
            for iteration in range(iter_max):
                if deadline is not None and perf_counter() > deadline:
                    break
                neighbours = []
                attempts = 0
                while len(neighbours) < 10 and attempts < 200:
                    attempts += 1
                    a, b = rng.sample(range(n), 2)
                    pair = tuple(sorted(
                        (current[a].item_id, current[b].item_id)
                    ))
                    if tabu.get(pair, -1) >= iteration:
                        continue
                    candidate = list(current)
                    candidate[a], candidate[b] = candidate[b], candidate[a]
                    neighbours.append((pair, candidate))
                if not neighbours:
                    break
                best_candidate = None
                for pair, candidate in neighbours:
                    if deadline is not None and perf_counter() > deadline:
                        break
                    result = skyline_heuristic(
                        candidate, strip_width, sheet_height, max_spread,
                        rotatable, deadline=deadline,
                    )
                    if record(result):
                        return True, best_placements, best_util
                    if best_candidate is None or result[1] > best_candidate[0]:
                        best_candidate = (result[1], pair, candidate)
                if best_candidate is None:
                    break
                _util, pair, current = best_candidate
                tabu[pair] = iteration + 3 * n
    return False, best_placements, best_util


def _lower_bound(items: list[Item], strip_width: int, rotatable: bool) -> int:
    lb1 = ceil(sum(item.area for item in items) / strip_width)
    if rotatable:
        return lb1
    lb2 = sum(item.height for item in items if item.width > strip_width / 2)
    lb3 = sum(item.height for item in items if item.width == strip_width / 2)
    return max(lb1, lb2 + ceil(lb3 / 2))


def idbs(
    instance: Instance,
    *,
    time_limit: float = 100.0,
    seed: int = 0,
    rotatable: bool = True,
    use_tabu: bool = True,
    max_rounds: int = 64,
) -> tuple[int | None, list[Placement] | None, float]:
    """Run Algorithm 2; return (best height, its placements, elapsed seconds)."""

    items = list(instance.items)
    strip_width = instance.strip_width
    rng = Random(seed)
    started = perf_counter()
    deadline = started + time_limit
    lb = _lower_bound(items, strip_width, rotatable)
    ub = max(lb + 1, round(lb * 1.1))
    iter_max = 1
    best_height: int | None = None
    best_placements: list[Placement] | None = None
    ub_found = False
    rounds = 0

    while (
        perf_counter() - started < time_limit
        and lb < ub
        and rounds < max_rounds
    ):
        rounds += 1
        temp_lb = lb
        while temp_lb < ub and perf_counter() - started < time_limit:
            sheet_height = (temp_lb + ub) // 2
            success, placements, _util = solver_2drp(
                items, strip_width, sheet_height, iter_max, rng, rotatable,
                use_tabu, deadline=deadline,
            )
            if success:
                ub = sheet_height
                ub_found = True
                assert placements is not None
                if best_height is None or sheet_height < best_height:
                    best_height, best_placements = sheet_height, placements
            else:
                temp_lb = sheet_height + 1
        if not ub_found:
            ub = round(ub * 1.1)
        iter_max *= 2

    return best_height, best_placements, perf_counter() - started
