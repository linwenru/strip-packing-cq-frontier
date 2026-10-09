"""Stage-3 baseline benchmark: every reproduced heuristic on one protocol.

Measures each reproduced algorithm family's original deterministic heuristic
on the 104 benchmark instances (C1-C7, BKW01-13, N/T1-7a-e) under the
stage-1 protocol — LB = max(ceil(total area / strip width), tallest item
height), gap = (height - LB) / LB — and streams long-format rows to
``docs/stage3-baselines.csv`` (resumable: completed (instance, algorithm)
cells are skipped).

Algorithm cells (``algorithm`` column):

- ``bf-LM`` / ``bf-TN`` / ``bf-SN``: Burke et al. 2004 best-fit with the
  three niche policies, tower-removal post-pass included (published variant).
- ``bbf-best288``: Asik & Ozcan 2009 BBF, best height over all 288
  combinations.
- ``twbf-best18``: Verstichel et al. 2013 TWBF, best over all 18
  ordering x policy combinations.
- ``ish-core``: Wei et al. 2017 BestFitPack core, best over the four
  RandomLS initial orderings (the deterministic starting point of RandomLS).
- ``sh-greedy``: Wei et al. 2011 skyline heuristic (rotatable variant, as
  reproduced in ``sky``), best over 6 sorting rules x 4 max_spread values —
  the No-Tabu first pass. The sheet height is set to the sum of the items'
  long sides, a trivially feasible strip height, so the largest spread value
  never constrains the packing; failed passes (possible under the tightest
  spreads) are skipped.
- ``ffdh`` / ``bfdh`` / ``sas`` / ``sasm`` / ``bfs`` / ``sc`` / ``scr``:
  the Ortmann et al. 2010 level family (``sas`` is the original Ntene &
  van Vuuren 2009 algorithm from ``lvl.solver``; ``scr`` is
  ``pack_sc(resort=True)``).
- ``fh``: Leung & Zhang 2011 FH. The first-improvement swap pass is O(n^2)
  evaluations and infeasible in plain Python for large n, so
  ``fast_heuristic``'s ``swap_budget`` caps it: full pass for n <= 250,
  5000 evaluations for 250 < n <= 500 (BKW11/BKW12; a full pass costs up to
  ~90 min on BKW12), and 500 evaluations for BKW13 (n=3152) — the budget
  the fh package's reproduction used to reach the paper's height 960..

Hard cost caps: ``bbf-best288`` and ``sh-greedy`` are skipped entirely on
instances with more than 500 items (BKW13, n=3152: a single greedy pass
alone exceeds 90 s there); those rows are absent from the CSV by design.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from math import ceil
from types import SimpleNamespace
from typing import Callable, Iterable

import bbf
import twbf
from burke_bf import load_instance
from burke_bf.model import Instance, Item, Placement, Policy
from burke_bf.solver import solve_policy, validate_solution
from fh import fast_heuristic
from ish import best_fit_pack
from ish.rls import SORT_RULES as ISH_SORT_RULES
from lvl import pack_bfdh, pack_bfs, pack_ffdh, pack_sasm, pack_sc
from lvl.solver import pack_sas
from sky import skyline_heuristic
from sky.idbs import SORT_RULES as SKY_SORT_RULES, _spread_values

DATA_SETS = ("data/c", "data/bkw", "data/nt")

# Instances with more items than this skip the bbf-best288 / sh-greedy cells.
LARGE_INSTANCE_N = 500
SKIP_ON_LARGE = ("bbf-best288", "sh-greedy")

# FH swap-pass budgets (see module docstring).
FH_FULL_SWAP_MAX_N = 250
FH_MID_SWAP_BUDGET = 5000
FH_LARGE_SWAP_BUDGET = 500

# One cell: instance -> (placements, height).
Runner = Callable[[Instance], tuple[tuple[Placement, ...], int]]


def _lower_bound(instance: Instance) -> int:
    area = sum(item.width * item.height for item in instance.items)
    tallest = max(item.height for item in instance.items)
    return max(ceil(area / instance.strip_width), tallest)


def _instance_paths() -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for data_set in DATA_SETS:
        for path in sorted(glob.glob(f"{data_set}/*.ins2D")):
            paths.append((data_set, path))
    return paths


def _as_tuple(
    result: tuple[list[Placement], int]
) -> tuple[tuple[Placement, ...], int]:
    placements, height = result
    return tuple(placements), height


def _run_bf(instance: Instance, policy: Policy) -> tuple[tuple[Placement, ...], int]:
    solution = solve_policy(instance, policy, postprocess=True)
    return solution.placements, solution.height


def _run_bbf(instance: Instance) -> tuple[tuple[Placement, ...], int]:
    best, _heights = bbf.solve(instance)
    return best.placements, best.height


def _run_twbf(instance: Instance) -> tuple[tuple[Placement, ...], int]:
    best, _heights = twbf.solve(instance)
    return best.placements, best.height


def _run_ish(instance: Instance) -> tuple[tuple[Placement, ...], int]:
    items = list(instance.items)
    best: tuple[int, tuple[Placement, ...]] | None = None
    for _name, key in ISH_SORT_RULES:
        sequence = sorted(items, key=key)
        placements, height = best_fit_pack(sequence, instance.strip_width)
        if best is None or height < best[0]:
            best = (height, tuple(placements))
    assert best is not None  # ISH_SORT_RULES is never empty
    height, placements = best
    return placements, height


def _run_sh(instance: Instance) -> tuple[tuple[Placement, ...], int]:
    items = list(instance.items)
    # Trivially feasible sheet height: stacking every item on its long side.
    sheet_height = sum(max(item.width, item.height) for item in items)
    best: tuple[int, tuple[Placement, ...]] | None = None
    for _name, key in SKY_SORT_RULES:
        sequence = sorted(items, key=key)
        for max_spread in _spread_values(items, sheet_height):
            placements, _util, success = skyline_heuristic(
                sequence, instance.strip_width, sheet_height, max_spread
            )
            if not success:
                continue  # tightest spreads can make a pass infeasible
            height = max(placement.top for placement in placements)
            if best is None or height < best[0]:
                best = (height, tuple(placements))
    if best is None:
        raise RuntimeError(f"sh-greedy: no successful pass on {instance.name}")
    height, placements = best
    return placements, height


def _fh_swap_budget(n_items: int) -> int | None:
    if n_items <= FH_FULL_SWAP_MAX_N:
        return None  # full O(n^2) first-improvement pass
    if n_items <= LARGE_INSTANCE_N:
        return FH_MID_SWAP_BUDGET
    return FH_LARGE_SWAP_BUDGET  # BKW13; reaches the paper's 960


def _run_fh(instance: Instance) -> tuple[tuple[Placement, ...], int]:
    return _as_tuple(
        fast_heuristic(
            list(instance.items),
            instance.strip_width,
            swap_budget=_fh_swap_budget(len(instance.items)),
        )
    )


def _run_lvl(
    instance: Instance, pack: Callable[[list[Item], int], tuple[list[Placement], int]]
) -> tuple[tuple[Placement, ...], int]:
    return _as_tuple(pack(list(instance.items), instance.strip_width))


ALGORITHMS: tuple[tuple[str, Runner], ...] = (
    ("bf-LM", lambda instance: _run_bf(instance, Policy.LEFTMOST)),
    ("bf-TN", lambda instance: _run_bf(instance, Policy.TALLEST_NEIGHBOUR)),
    ("bf-SN", lambda instance: _run_bf(instance, Policy.SHORTEST_NEIGHBOUR)),
    ("bbf-best288", _run_bbf),
    ("twbf-best18", _run_twbf),
    ("ish-core", _run_ish),
    ("sh-greedy", _run_sh),
    ("ffdh", lambda instance: _run_lvl(instance, pack_ffdh)),
    ("bfdh", lambda instance: _run_lvl(instance, pack_bfdh)),
    ("sas", lambda instance: _run_lvl(instance, pack_sas)),
    ("sasm", lambda instance: _run_lvl(instance, pack_sasm)),
    ("bfs", lambda instance: _run_lvl(instance, pack_bfs)),
    ("sc", lambda instance: _run_lvl(instance, pack_sc)),
    (
        "scr",
        lambda instance: _as_tuple(
            pack_sc(list(instance.items), instance.strip_width, resort=True)
        ),
    ),
    ("fh", _run_fh),
)


def _validate(
    instance: Instance, placements: tuple[Placement, ...], height: int
) -> None:
    validate_solution(
        instance, SimpleNamespace(placements=placements, height=height)
    )


def _done_cells(rows_path: str) -> set[tuple[str, str]]:
    try:
        with open(rows_path, newline="") as handle:
            return {
                (row["instance"], row["algorithm"])
                for row in csv.DictReader(handle)
            }
    except FileNotFoundError:
        return set()


def _run(paths: list[tuple[str, str]], rows_path: str) -> None:
    done = _done_cells(rows_path)
    fresh = not done
    total = len(paths) * len(ALGORITHMS)
    completed = len(done)
    with open(rows_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(
                [
                    "instance",
                    "data_set",
                    "n_items",
                    "algorithm",
                    "height",
                    "lower_bound",
                    "gap_pct",
                ]
            )
        for data_set, path in paths:
            instance = load_instance(path)
            lower_bound = _lower_bound(instance)
            n_items = len(instance.items)
            start = time.perf_counter()
            skipped = 0
            for algorithm, runner in ALGORITHMS:
                if (instance.name, algorithm) in done:
                    skipped += 1
                    continue
                if n_items > LARGE_INSTANCE_N and algorithm in SKIP_ON_LARGE:
                    skipped += 1
                    continue  # hard cost cap; row absent by design
                cell_start = time.perf_counter()
                placements, height = runner(instance)
                _validate(instance, placements, height)
                writer.writerow(
                    [
                        instance.name,
                        data_set,
                        n_items,
                        algorithm,
                        height,
                        lower_bound,
                        f"{100.0 * (height - lower_bound) / lower_bound:.4f}",
                    ]
                )
                completed += 1
                print(
                    f"  {instance.name} {algorithm} h={height}"
                    f" ({time.perf_counter() - cell_start:.1f}s)",
                    flush=True,
                )
            handle.flush()
            print(
                f"[{completed}/{total}] {instance.name}"
                f" ({time.perf_counter() - start:.1f}s, {skipped} cached)",
                flush=True,
            )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N instances (smoke test)",
    )
    parser.add_argument("--out-dir", default="docs", help="output directory")
    parser.add_argument(
        "--runs-csv",
        default="stage3-baselines.csv",
        help="long-format results file name inside --out-dir",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows_path = f"{args.out_dir}/{args.runs_csv}"
    paths = _instance_paths()
    if args.limit is not None:
        paths = paths[: args.limit]

    _run(paths, rows_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
