"""Stage-1 factorial component experiment over the CCH component space.

Runs all 6 orderings x 6 selection rules x 6 placement policies (216
configurations, one deterministic pass each) over the 104 benchmark
instances (C1-C7, BKW01-13, N/T1-7a-e), streams long-format results to
``docs/stage1-runs.csv`` (resumable: completed cells are skipped), then
derives per-configuration means (``docs/stage1-configs.csv``), marginal
main effects (``docs/stage1-marginals.csv``) and pairwise paired-t
comparisons between the levels of each component axis
(``docs/stage1-pairwise.csv``).

The relative gap of a packing is (height - LB) / LB with
LB = max(ceil(total area / strip width), tallest item height).
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from dataclasses import dataclass
from itertools import product
from math import ceil, sqrt
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Iterable

from burke_bf import load_instance
from burke_bf.solver import validate_solution

from .model import Config, Ordering, PlacementPolicy, Selection
from .solver import solve_config

DATA_SETS = ("data/c", "data/bkw", "data/nt")

# Two-sided critical values, df = 103 (104 instances).
T_95 = 1.984
T_99 = 2.626


@dataclass(frozen=True, slots=True)
class _Run:
    instance: str
    data_set: str
    n_items: int
    ordering: str
    selection: str
    placement: str
    height: int
    lower_bound: int
    gap_pct: float


def _lower_bound(instance) -> int:
    area = sum(item.width * item.height for item in instance.items)
    tallest = max(item.height for item in instance.items)
    return max(ceil(area / instance.strip_width), tallest)


def _instance_paths() -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for data_set in DATA_SETS:
        for path in sorted(glob.glob(f"{data_set}/*.ins2D")):
            paths.append((data_set, path))
    return paths


def _self_check(
    instance,
    orderings: list[Ordering],
    selections: list[Selection],
    placements: list[PlacementPolicy],
    tower_removal: bool,
    rotation_rule: bool,
    vertical_niche: bool,
) -> None:
    """Geometry-validate all configurations of the run on one small instance."""

    for ordering, selection, placement in product(orderings, selections, placements):
        config = Config(
            ordering=ordering,
            selection=selection,
            placement=placement,
            tower_removal=tower_removal,
            rotation_rule=rotation_rule,
            vertical_niche=vertical_niche,
        )
        placements_out, skyline, _ = solve_config(instance, config)
        validate_solution(
            instance,
            SimpleNamespace(
                placements=tuple(placements_out), height=max(skyline)
            ),
        )


def _done_cells(rows_path: str) -> set[tuple[str, str, str, str]]:
    try:
        with open(rows_path, newline="") as handle:
            return {
                (row["instance"], row["ordering"], row["selection"], row["placement"])
                for row in csv.DictReader(handle)
            }
    except FileNotFoundError:
        return set()


def _run(
    paths: list[tuple[str, str]],
    rows_path: str,
    orderings: list[Ordering],
    selections: list[Selection],
    placements: list[PlacementPolicy],
    tower_removal: bool,
    rotation_rule: bool,
    vertical_niche: bool,
) -> None:
    done = _done_cells(rows_path)
    fresh = not done
    n_configs = len(orderings) * len(selections) * len(placements)
    total = len(paths) * n_configs
    completed = len(done)
    with open(rows_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(
                [
                    "instance",
                    "data_set",
                    "n_items",
                    "ordering",
                    "selection",
                    "placement",
                    "height",
                    "lower_bound",
                    "gap_pct",
                ]
            )
        for data_set, path in paths:
            instance = load_instance(path)
            lower_bound = _lower_bound(instance)
            start = time.perf_counter()
            skipped = 0
            for ordering, selection, placement in product(
                orderings, selections, placements
            ):
                key = (instance.name, ordering.value, selection.value, placement.value)
                if key in done:
                    skipped += 1
                    continue
                config = Config(
                    ordering=ordering,
                    selection=selection,
                    placement=placement,
                    tower_removal=tower_removal,
                    rotation_rule=rotation_rule,
                    vertical_niche=vertical_niche,
                )
                placements_out, skyline, _ = solve_config(instance, config)
                height = max(skyline)
                writer.writerow(
                    [
                        instance.name,
                        data_set,
                        len(instance.items),
                        ordering.value,
                        selection.value,
                        placement.value,
                        height,
                        lower_bound,
                        f"{100.0 * (height - lower_bound) / lower_bound:.4f}",
                    ]
                )
            completed += n_configs - skipped
            handle.flush()
            print(
                f"[{completed}/{total}] {instance.name}"
                f" ({time.perf_counter() - start:.1f}s, {skipped} cached)",
                flush=True,
            )


def _read_runs(rows_path: str) -> list[_Run]:
    with open(rows_path, newline="") as handle:
        return [
            _Run(
                instance=row["instance"],
                data_set=row["data_set"],
                n_items=int(row["n_items"]),
                ordering=row["ordering"],
                selection=row["selection"],
                placement=row["placement"],
                height=int(row["height"]),
                lower_bound=int(row["lower_bound"]),
                gap_pct=float(row["gap_pct"]),
            )
            for row in csv.DictReader(handle)
        ]


def _write_config_means(runs: list[_Run], path: str) -> None:
    cells: dict[tuple[str, str, str], list[_Run]] = {}
    for run in runs:
        cells.setdefault((run.ordering, run.selection, run.placement), []).append(run)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ordering",
                "selection",
                "placement",
                "mean_gap_pct",
                "mean_gap_c",
                "mean_gap_bkw",
                "mean_gap_nt",
                "instances",
            ]
        )
        rows = []
        for (ordering, selection, placement), cell in cells.items():
            by_set = {
                name: [r.gap_pct for r in cell if r.data_set == name]
                for name in DATA_SETS
            }
            rows.append(
                (
                    mean(r.gap_pct for r in cell),
                    ordering,
                    selection,
                    placement,
                    *(mean(gaps) if gaps else "" for gaps in by_set.values()),
                    len(cell),
                )
            )
        for row in sorted(rows):
            writer.writerow(
                (row[1], row[2], row[3], f"{row[0]:.4f}", *row[4:])
            )


def _write_main_effects(
    runs: list[_Run], marginals_path: str, pairwise_path: str
) -> None:
    # Axis name -> (key index in the gap lookup, level values).
    axes = (
        ("ordering", 1, [level.value for level in Ordering]),
        ("selection", 2, [level.value for level in Selection]),
        ("placement", 3, [level.value for level in PlacementPolicy]),
    )
    # gap lookup: (instance, ordering, selection, placement) -> gap
    gap = {
        (run.instance, run.ordering, run.selection, run.placement): run.gap_pct
        for run in runs
    }
    instances = sorted({run.instance for run in runs})

    with open(marginals_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "level", "marginal_gap_pct", "cells"])
        for axis, key_index, levels in axes:
            for level in levels:
                values = [
                    value for key, value in gap.items() if key[key_index] == level
                ]
                writer.writerow([axis, level, f"{mean(values):.4f}", len(values)])

    with open(pairwise_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["axis", "level_a", "level_b", "marginal_a", "marginal_b",
             "mean_diff", "t_stat", "sig"]
        )
        for axis, key_index, levels in axes:
            other_indexes = tuple(i for _, i, _ in axes if i != key_index)
            other_levels = [axes[i - 1][2] for i in other_indexes]
            cells = list(product(*other_levels))
            marginals = {
                level: mean(
                    value for key, value in gap.items() if key[key_index] == level
                )
                for level in levels
            }
            for index_a, level_a in enumerate(levels):
                for level_b in levels[index_a + 1 :]:
                    diffs = []
                    for instance in instances:
                        per_diff = []
                        for cell in cells:
                            key_a: list[str] = [instance, "", "", ""]
                            key_b: list[str] = [instance, "", "", ""]
                            key_a[key_index] = level_a
                            key_b[key_index] = level_b
                            for other_index, value in zip(other_indexes, cell):
                                key_a[other_index] = value
                                key_b[other_index] = value
                            per_diff.append(gap[tuple(key_a)] - gap[tuple(key_b)])
                        diffs.append(mean(per_diff))
                    mean_diff = mean(diffs)
                    spread = stdev(diffs)
                    t_stat = (
                        mean_diff / (spread / sqrt(len(diffs)))
                        if spread > 0
                        else 0.0
                    )
                    sig = (
                        "**"
                        if abs(t_stat) > T_99
                        else "*" if abs(t_stat) > T_95 else ""
                    )
                    writer.writerow(
                        [
                            axis,
                            level_a,
                            level_b,
                            f"{marginals[level_a]:.4f}",
                            f"{marginals[level_b]:.4f}",
                            f"{mean_diff:.4f}",
                            f"{t_stat:.2f}",
                            sig,
                        ]
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
        default="stage1-runs.csv",
        help="long-format results file name inside --out-dir",
    )
    parser.add_argument(
        "--tower-removal",
        action="store_true",
        help="enable Burke's tower-removal post-pass (stage-2 run)",
    )
    parser.add_argument(
        "--rotation-rule",
        action="store_true",
        help="enable the TWBF-style 2n configuration table (stage-2 run; "
        "restricts orderings to width/height/area and selections to "
        "widest-fit/first-fit/tre)",
    )
    parser.add_argument(
        "--vertical-niche",
        action="store_true",
        help="add BBF's vertical niche as an alternative placement site "
        "(stage-2 run, D1 axis)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="skip the stage-1 summary statistics (stage-2 runs)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    orderings = list(Ordering)
    selections = list(Selection)
    if args.rotation_rule:
        orderings = [Ordering.WIDTH, Ordering.HEIGHT, Ordering.AREA]
        selections = [Selection.WIDEST_FIT, Selection.FIRST_FIT, Selection.TRE]
    placements = list(PlacementPolicy)

    rows_path = f"{args.out_dir}/{args.runs_csv}"
    paths = _instance_paths()
    if args.limit is not None:
        paths = paths[: args.limit]

    n_configs = len(orderings) * len(selections) * len(placements)
    print(f"self-check: {n_configs} configurations on c1-p1 ...", flush=True)
    _self_check(
        load_instance("data/c/c1-p1.ins2D"),
        orderings,
        selections,
        placements,
        args.tower_removal,
        args.rotation_rule,
        args.vertical_niche,
    )
    print("self-check passed", flush=True)

    _run(
        paths,
        rows_path,
        orderings,
        selections,
        placements,
        args.tower_removal,
        args.rotation_rule,
        args.vertical_niche,
    )

    if args.no_stats:
        return 0
    runs = _read_runs(rows_path)
    expected = 104 * n_configs
    if len(runs) != expected:
        print(
            f"note: {len(runs)}/{expected} cells present; statistics cover "
            "the completed cells only",
            flush=True,
        )
    _write_config_means(runs, f"{args.out_dir}/stage1-configs.csv")
    _write_main_effects(
        runs,
        f"{args.out_dir}/stage1-marginals.csv",
        f"{args.out_dir}/stage1-pairwise.csv",
    )
    print("wrote stage1-configs.csv, stage1-marginals.csv, stage1-pairwise.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
