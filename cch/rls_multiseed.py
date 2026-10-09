"""Multi-seed RLS experiment: 3 cores x 34 instances x 10 seeds x 20 s.

External-review item: the original Table-5 evidence was single-seed 20 s.
This runner reports the mean +/- sd across 10 independent seeds per core,
plus the best-of-seeds per instance, streamed to
docs/stage3-rls-multiseed.csv with resume support.

Usage: python3 -m cch.rls_multiseed [--procs N]
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from multiprocessing import Pool
from pathlib import Path

from burke_bf import Instance, load_instance

from .model import Config, Ordering, PlacementPolicy, Selection
from .shell import random_ls

CORES = {
    "classic-bf": Config(Ordering.WIDTH, Selection.WIDEST_FIT, PlacementPolicy.LM),
    "stage1-best": Config(Ordering.PERIMETER, Selection.FIRST_FIT, PlacementPolicy.TN),
    "stage2-best": Config(
        Ordering.PERIMETER, Selection.FITNESS_NUMBER, PlacementPolicy.SN,
        tower_removal=True,
    ),
}
SEEDS = list(range(1, 11))
TIME_LIMIT = 20.0
OUT = Path("docs/stage3-rls-multiseed.csv")
FIELDS = ["instance", "core", "seed", "initial", "best", "lb", "gap_pct", "elapsed_s", "packs"]


def _lower_bound(instance: Instance) -> int:
    area = sum(item.width * item.height for item in instance.items)
    return max(math.ceil(area / instance.strip_width), 0)


def _tasks() -> list[tuple[str, str, int]]:
    paths = sorted(glob.glob("data/c/c*-p*.ins2D")) + sorted(glob.glob("data/bkw/BKW*.ins2D"))
    done = set()
    if OUT.exists():
        with open(OUT, newline="") as handle:
            for row in csv.DictReader(handle):
                done.add((row["instance"], row["core"], int(row["seed"])))
    tasks = []
    for path in paths:
        name = load_instance(path).name
        for core in CORES:
            for seed in SEEDS:
                if (name, core, seed) not in done:
                    tasks.append((path, core, seed))
    return tasks


def _run(task: tuple[str, str, int]) -> dict:
    path, core_name, seed = task
    instance = load_instance(path)
    initial, best, _pl, elapsed, packs = random_ls(
        instance, CORES[core_name], time_limit=TIME_LIMIT, seed=seed
    )
    lb = _lower_bound(instance)
    return {
        "instance": instance.name,
        "core": core_name,
        "seed": seed,
        "initial": initial,
        "best": best,
        "lb": lb,
        "gap_pct": f"{100.0 * (best - lb) / lb:.4f}",
        "elapsed_s": f"{elapsed:.1f}",
        "packs": packs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=6)
    args = parser.parse_args()
    tasks = _tasks()
    print(f"{len(tasks)} cells to run (pool={args.procs})")
    first = not OUT.exists()
    with open(OUT, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if first:
            writer.writeheader()
        with Pool(args.procs) as pool:
            for i, row in enumerate(pool.imap_unordered(_run, tasks), 1):
                writer.writerow(row)
                handle.flush()
                if i % 50 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} done", flush=True)


if __name__ == "__main__":
    main()
