"""Mechanism measurements: why TN and widest-fit work (stage-4 analyses).

An instrumented replay of the engine's packing loop records, per run: the
number of gap-raising events, the buried waste (total area raised over and
thus rendered unreachable), the exact-fill rate, and the residual sliver
widths. Configurations: width ordering x {widest-fit, first-fit,
fitness-number, max-area} x the six placement policies, on the 104
benchmark instances.

    python3 -m cch.mechanisms

writes docs/stage4-mechanisms.csv and prints the summary tables.
"""

from __future__ import annotations

import csv
import sys
from math import inf
from statistics import mean, stdev
from math import sqrt

from burke_bf import load_instance

from .experiment import _instance_paths, _lower_bound
from .model import Config, Ordering, PlacementPolicy, Selection
from .solver import (
    _lowest_gap,
    _neighbours,
    _place_on_left,
    _raise,
    _select,
    _sort,
)

SELECTIONS = (
    Selection.WIDEST_FIT,
    Selection.FIRST_FIT,
    Selection.FITNESS_NUMBER,
    Selection.MAX_AREA,
)


def measure(instance, config: Config, tower_removal: bool = False):
    """Pack; return (raises, buried, exact_fills, placements, residuals, skyline).

    With ``tower_removal`` the tower-removal post-pass is applied, mirroring
    ``solver._reduce_towers``: the highest placement is rotated 90 degrees
    and reinserted through the lowest-gap/raise loop while that strictly
    improves the height (the pass stops when the highest placement is not
    portrait or cannot rotate within the strip). Raises and buried waste of
    a reinsertion are accounted in the same counters, but only for accepted
    moves; a rejected trial's raises are rolled back.
    ``placements`` is a list of (x, y, width, height, rectangle).
    """

    remaining = _sort(list(instance.items), config.ordering, instance.strip_width, True)
    skyline = [0] * instance.strip_width
    raises = 0
    buried = 0
    exact = 0
    placed: list[tuple[int, int, int, int, object]] = []
    residuals: list[int] = []

    def place_one(remaining) -> bool:
        """Place the first selectable rectangle; return False if none fits."""
        nonlocal raises, buried, exact
        gap = _lowest_gap(skyline)
        found = _select(config, remaining, skyline, gap)
        if found is None:
            left, right = _neighbours(skyline, gap)
            raises += 1
            buried += gap.width * (min(left, right) - gap.y)
            _raise(skyline, gap)
            return False
        index, rectangle, width, height = found
        if config.selection is Selection.FITNESS_NUMBER:
            left, right = _neighbours(skyline, gap)
            lh = left - gap.y if left != inf else inf
            rh = right - gap.y if right != inf else inf
            fit_left = (width == gap.width) + (height == lh) + (
                width == gap.width and height == rh
            )
            fit_right = (width == gap.width) + (height == rh) + (
                width == gap.width and height == lh
            )
            on_left = fit_left > fit_right if fit_left != fit_right else lh >= rh
        else:
            on_left = _place_on_left(config.placement, skyline, gap, gap.y + height)
        x = gap.x if on_left else gap.right - width
        skyline[x : x + width] = [gap.y + height] * width
        if width == gap.width:
            exact += 1
        else:
            residuals.append(gap.width - width)
        placed.append((x, gap.y, width, height, rectangle))
        del remaining[index]
        return True

    while remaining:
        place_one(remaining)

    if tower_removal:
        # Mirrors solver._reduce_towers exactly: pick the highest placement
        # overall (ties: earliest placed); stop when it is not portrait or
        # cannot rotate within the strip; reinsert the rotated rectangle
        # through the lowest-gap/raise loop at the placement policy's side;
        # keep the move only on strict improvement. Raises/buried of a trial
        # reinsertion are committed to the counters only when the move is
        # accepted (a rejected trial's raises are rolled back).
        while True:
            top_index, top_entry = max(
                enumerate(placed), key=lambda pair: pair[1][1] + pair[1][3]
            )
            x, y, w, h, rectangle = top_entry
            if w >= h or h > instance.strip_width:
                break
            old_height = max(skyline)
            old_skyline = skyline.copy()
            del placed[top_index]
            skyline[x : x + w] = [value - h for value in skyline[x : x + w]]
            required, r_height = h, w  # rotated dimensions
            trial_raises = 0
            trial_buried = 0
            while True:
                gap = _lowest_gap(skyline)
                if required <= gap.width:
                    on_left = _place_on_left(
                        config.placement, skyline, gap, gap.y + r_height
                    )
                    nx = gap.x if on_left else gap.right - required
                    skyline[nx : nx + required] = [gap.y + r_height] * required
                    placed.append(
                        (nx, gap.y, required, r_height,
                         type(rectangle)(rectangle.item, required, r_height))
                    )
                    break
                left, right = _neighbours(skyline, gap)
                trial_raises += 1
                trial_buried += gap.width * (min(left, right) - gap.y)
                _raise(skyline, gap)
            if max(skyline) >= old_height:
                placed.pop()
                placed.insert(top_index, top_entry)
                skyline = old_skyline
                break
            raises += trial_raises
            buried += trial_buried

    return raises, buried, exact, placed, residuals, skyline


def _t(diffs: list[float]) -> float:
    return mean(diffs) / (stdev(diffs) / sqrt(len(diffs))) if stdev(diffs) > 0 else 0.0


def main() -> int:
    paths = _instance_paths()
    rows = []
    for data_set, path in paths:
        instance = load_instance(path)
        lower_bound = _lower_bound(instance)
        for selection in SELECTIONS:
            for placement in PlacementPolicy:
                config = Config(
                    ordering=Ordering.WIDTH,
                    selection=selection,
                    placement=placement,
                )
                raises, buried, exact, placed, residuals, skyline = measure(
                    instance, config
                )
                height = max(skyline)
                strip_area = instance.strip_width
                # Height excess decomposition: H*W = area + buried + top waste.
                top_waste = height * strip_area - instance.total_area - buried
                rows.append(
                    {
                        "instance": instance.name,
                        "selection": selection.value,
                        "placement": placement.value,
                        "raises": raises,
                        "buried_area": buried,
                        "buried_pct": 100.0 * buried / (lower_bound * strip_area),
                        "top_waste_pct": 100.0 * top_waste / (lower_bound * strip_area),
                        "exact_fill_pct": 100.0 * exact / len(placed),
                        "mean_residual": mean(residuals) if residuals else 0.0,
                    }
                )
    with open("docs/stage4-mechanisms.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("== M1: 放置策略与浪费机制（全选件规则平均）==")
    for placement in PlacementPolicy:
        sub = [r for r in rows if r["placement"] == placement.value]
        print(
            f"{placement.value:6s} raises={mean(float(r['raises']) for r in sub):7.2f}  "
            f"buried%={mean(float(r['buried_pct']) for r in sub):6.3f}"
        )
    print("\n配对 t（buried_pct，逐实例）:")
    for a, b in (("TN", "LM"), ("TN", "SN"), ("LM", "SN"), ("TN", "MinD")):
        diffs = [
            next(r for r in rows if r["instance"] == i and r["placement"] == a
                 and r["selection"] == s)["buried_pct"]
            - next(r for r in rows if r["instance"] == i and r["placement"] == b
                   and r["selection"] == s)["buried_pct"]
            for i in {r["instance"] for r in rows}
            for s in {r["selection"] for r in rows}
        ]
        diffs = [float(d) for d in diffs]
        print(f"  {a} − {b}: mean={mean(diffs):+.3f}, t={_t(diffs):.2f}")

    print("\n== M2: 选件规则与浪费机制（全放置策略平均）==")
    for selection in SELECTIONS:
        sub = [r for r in rows if r["selection"] == selection.value]
        print(
            f"{selection.value:15s} raises={mean(float(r['raises']) for r in sub):7.2f}  "
            f"buried%={mean(float(r['buried_pct']) for r in sub):6.3f}  "
            f"top%={mean(float(r['top_waste_pct']) for r in sub):6.3f}  "
            f"exact%={mean(float(r['exact_fill_pct']) for r in sub):6.2f}  "
            f"resid={mean(float(r['mean_residual']) for r in sub):5.2f}"
        )
    print("\n配对 t（buried_pct，逐实例）:")
    for a, b in (("widest-fit", "first-fit"), ("widest-fit", "max-area")):
        diffs = [
            next(r for r in rows if r["instance"] == i and r["selection"] == a
                 and r["placement"] == p)["buried_pct"]
            - next(r for r in rows if r["instance"] == i and r["selection"] == b
                   and r["placement"] == p)["buried_pct"]
            for i in {r["instance"] for r in rows}
            for p in {r["placement"] for r in rows}
        ]
        diffs = [float(d) for d in diffs]
        print(f"  {a} − {b}: mean={mean(diffs):+.3f}, t={_t(diffs):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
