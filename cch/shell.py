"""Generic random local search shell (D6=rls) over the CCH engine.

Mirrors ISH's parameter-free RandomLS (Wei et al. 2017, Algorithm 2), but
with any engine configuration as the packing core: the engine's six
orderings provide the initial sequences (re-ordered by returned height),
then each gets n single-swap hill-climbing iterations per outer round
(accept when not worse), halting early at the area lower bound. Used to
quantify the shell gain of any component configuration.

As a script it runs the shell-gain experiment over selected configurations:

    python3 -m cch.shell [--time-limit SEC] [--seed N] [--limit N]

writes docs/stage3-rls-runs.csv (resumable).
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from math import ceil, sqrt
from random import Random
from time import perf_counter
from types import SimpleNamespace
from typing import Iterable

from burke_bf import load_instance
from burke_bf.model import Instance, Item, Placement
from burke_bf.solver import validate_solution

from .model import Config, Ordering, PlacementPolicy, Selection
from .solver import solve_config

DATA_SETS = ("data/c", "data/bkw")

# The three configurations whose shell gain is measured: classic BF, the
# stage-1 best, and the stage-2 best (tower removal enabled).
SHELL_CONFIGS = (
    ("bf-classic", Config(selection=Selection.WIDEST_FIT)),
    (
        "stage1-best",
        Config(
            ordering=Ordering.PERIMETER,
            selection=Selection.FIRST_FIT,
            placement=PlacementPolicy.TN,
        ),
    ),
    (
        "stage2-best",
        Config(
            ordering=Ordering.PERIMETER,
            selection=Selection.FITNESS_NUMBER,
            placement=PlacementPolicy.SN,
            tower_removal=True,
        ),
    ),
)


def _sort_key(ordering: Ordering, item: Item) -> tuple[float, ...]:
    """Ordering keys on the item's normalised dimensions (w >= h)."""

    wide = max(item.width, item.height)
    tall = min(item.width, item.height)
    if ordering is Ordering.WIDTH or ordering is Ordering.MAXSIDE:
        return (-wide, -tall)
    if ordering is Ordering.HEIGHT:
        return (-tall, -wide)
    if ordering is Ordering.AREA:
        return (-wide * tall, -wide)
    if ordering is Ordering.PERIMETER:
        return (-2 * (wide + tall), -wide)
    return (-(sqrt(wide**2 + tall**2) + wide + tall), -wide)


def random_ls(
    instance: Instance,
    config: Config,
    *,
    time_limit: float = 60.0,
    seed: int = 1,
) -> tuple[int, int, list[Placement], float, int]:
    """Run the shell; return (initial, best, placements, elapsed, #packs)."""

    items = list(instance.items)
    rng = Random(seed)
    started = perf_counter()
    n = len(items)
    lower_bound = ceil(instance.total_area / instance.strip_width)
    evaluations = 0

    def pack(sequence: list[Item]) -> tuple[list[Placement], int]:
        nonlocal evaluations
        evaluations += 1
        placements, skyline, _ = solve_config(instance, config, sequence=sequence)
        return placements, max(skyline)

    initial: list[tuple[int, str, list[Item], list[Placement]]] = []
    for ordering in Ordering:
        sequence = sorted(items, key=lambda item: _sort_key(ordering, item))
        placements, height = pack(sequence)
        initial.append((height, ordering.value, sequence, placements))
    initial.sort(key=lambda entry: (entry[0], entry[1]))

    initial_best, best_height, best_placements = (
        initial[0][0],
        initial[0][0],
        initial[0][3],
    )
    if best_height == lower_bound:
        return initial_best, best_height, best_placements, perf_counter() - started, evaluations

    while perf_counter() - started < time_limit:
        for _height, _name, sequence, _placements in initial:
            if perf_counter() - started > time_limit:
                break
            current = sequence
            placements, height = pack(current)
            if height < best_height:
                best_height, best_placements = height, placements
            if height == lower_bound:
                return initial_best, best_height, best_placements, perf_counter() - started, evaluations
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
                        return initial_best, best_height, best_placements, perf_counter() - started, evaluations

    return initial_best, best_height, best_placements, perf_counter() - started, evaluations


def _lower_bound(instance: Instance) -> int:
    tallest = max(item.height for item in instance.items)
    return max(ceil(instance.total_area / instance.strip_width), tallest)


def _instance_paths() -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for data_set in DATA_SETS:
        for path in sorted(glob.glob(f"{data_set}/*.ins2D")):
            paths.append((data_set, path))
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="docs")
    parser.add_argument("--runs-csv", default="stage3-rls-runs.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows_path = f"{args.out_dir}/{args.runs_csv}"
    done: set[tuple[str, str]] = set()
    try:
        with open(rows_path, newline="") as handle:
            done = {(row["instance"], row["config"]) for row in csv.DictReader(handle)}
    except FileNotFoundError:
        pass

    paths = _instance_paths()
    if args.limit is not None:
        paths = paths[: args.limit]
    total = len(paths) * len(SHELL_CONFIGS)

    with open(rows_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not done:
            writer.writerow(
                [
                    "instance",
                    "data_set",
                    "n_items",
                    "config",
                    "lower_bound",
                    "initial_height",
                    "rls_height",
                    "gap_initial_pct",
                    "gap_rls_pct",
                    "elapsed_s",
                    "evaluations",
                ]
            )
        completed = len(done)
        for data_set, path in paths:
            instance = load_instance(path)
            lower_bound = _lower_bound(instance)
            start = perf_counter()
            for config_name, config in SHELL_CONFIGS:
                if (instance.name, config_name) in done:
                    continue
                initial, best, placements, elapsed, evaluations = random_ls(
                    instance, config, time_limit=args.time_limit, seed=args.seed
                )
                validate_solution(
                    instance,
                    SimpleNamespace(placements=tuple(placements), height=best),
                )
                writer.writerow(
                    [
                        instance.name,
                        data_set,
                        len(instance.items),
                        config_name,
                        lower_bound,
                        initial,
                        best,
                        f"{100.0 * (initial - lower_bound) / lower_bound:.4f}",
                        f"{100.0 * (best - lower_bound) / lower_bound:.4f}",
                        f"{elapsed:.1f}",
                        evaluations,
                    ]
                )
                completed += 1
            handle.flush()
            print(
                f"[{completed}/{total}] {instance.name}"
                f" ({perf_counter() - start:.1f}s)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
