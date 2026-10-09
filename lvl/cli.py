"""Command-line runner for the level-heuristic reproduction experiment.

Runs FFDH, BFDH, SASm, BFS, SC and SCR on the C (Hopper & Turton 2001) and
N/T (Hopper 2000) sets and compares per-category gap against Table 2 of
Ortmann, Ntene and van Vuuren (2010).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from burke_bf import load_instance

from .solver import pack_bfdh, pack_bfs, pack_ffdh, pack_sasm, pack_sc

ALGORITHMS = {
    "FFDH": pack_ffdh,
    "BFDH": pack_bfdh,
    "SASm": pack_sasm,
    "BFS": pack_bfs,
    "SC": lambda items, w: pack_sc(items, w, resort=False),
    "SCR": lambda items, w: pack_sc(items, w, resort=True),
}

# Ortmann et al. (2010) Table 2: category -> algorithm -> gap (%).
PAPER_TABLE_2 = {
    "C1": {"FFDH": 40.0, "BFDH": 36.7, "SAS": 36.7, "SASm": 28.3, "BFS": 15.0, "SC": 13.3, "SCR": 10.0},
    "C2": {"FFDH": 15.6, "BFDH": 15.6, "SAS": 20.0, "SASm": 11.1, "BFS": 11.1, "SC": 8.9, "SCR": 11.1},
    "C3": {"FFDH": 25.6, "BFDH": 25.6, "SAS": 23.3, "SASm": 16.7, "BFS": 17.8, "SC": 14.4, "SCR": 15.6},
    "C4": {"FFDH": 26.7, "BFDH": 26.7, "SAS": 20.6, "SASm": 15.6, "BFS": 11.7, "SC": 6.1, "SCR": 9.4},
    "C5": {"FFDH": 16.3, "BFDH": 16.3, "SAS": 12.2, "SASm": 11.1, "BFS": 9.6, "SC": 7.0, "SCR": 8.1},
    "C6": {"FFDH": 17.2, "BFDH": 16.7, "SAS": 13.3, "SASm": 11.1, "BFS": 6.1, "SC": 5.3, "SCR": 5.8},
    "C7": {"FFDH": 13.5, "BFDH": 13.5, "SAS": 10.8, "SASm": 9.0, "BFS": 4.9, "SC": 3.9, "SCR": 4.9},
    "T1": {"FFDH": 46.5, "BFDH": 46.5, "SAS": 41.2, "SASm": 41.2, "BFS": 36.8, "SC": 30.3, "SCR": 28.5},
    "T2": {"FFDH": 40.8, "BFDH": 40.8, "SAS": 34.4, "SASm": 31.2, "BFS": 27.0, "SC": 21.8, "SCR": 19.4},
    "T3": {"FFDH": 40.7, "BFDH": 40.6, "SAS": 34.6, "SASm": 34.6, "BFS": 22.0, "SC": 15.4, "SCR": 15.7},
    "T4": {"FFDH": 27.5, "BFDH": 27.5, "SAS": 21.9, "SASm": 23.5, "BFS": 13.9, "SC": 8.9, "SCR": 11.9},
    "T5": {"FFDH": 27.2, "BFDH": 27.2, "SAS": 17.4, "SASm": 17.4, "BFS": 12.2, "SC": 7.8, "SCR": 8.2},
    "T6": {"FFDH": 31.7, "BFDH": 31.7, "SAS": 16.6, "SASm": 16.7, "BFS": 9.9, "SC": 6.0, "SCR": 8.0},
    "T7": {"FFDH": 27.4, "BFDH": 27.4, "SAS": 12.5, "SASm": 10.8, "BFS": 6.3, "SC": 5.1, "SCR": 5.6},
    "N1": {"FFDH": 37.5, "BFDH": 35.9, "SAS": 27.1, "SASm": 27.1, "BFS": 23.9, "SC": 20.2, "SCR": 20.5},
    "N2": {"FFDH": 33.1, "BFDH": 33.1, "SAS": 34.7, "SASm": 30.6, "BFS": 19.5, "SC": 18.5, "SCR": 17.3},
    "N3": {"FFDH": 36.8, "BFDH": 36.8, "SAS": 29.8, "SASm": 28.6, "BFS": 20.3, "SC": 16.0, "SCR": 17.2},
    "N4": {"FFDH": 28.6, "BFDH": 28.6, "SAS": 22.4, "SASm": 22.5, "BFS": 14.3, "SC": 9.6, "SCR": 12.8},
    "N5": {"FFDH": 28.2, "BFDH": 28.2, "SAS": 18.9, "SASm": 18.9, "BFS": 12.3, "SC": 8.2, "SCR": 8.6},
    "N6": {"FFDH": 28.2, "BFDH": 28.1, "SAS": 17.7, "SASm": 17.7, "BFS": 9.8, "SC": 5.6, "SCR": 8.5},
    "N7": {"FFDH": 23.9, "BFDH": 23.9, "SAS": 14.8, "SASm": 14.4, "BFS": 5.6, "SC": 4.5, "SCR": 5.7},
}


def _category(name: str) -> str:
    # C1P2 -> C1, N2c -> N2, T3d -> T3
    return f"{name[0]}{name[1]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run level-packing heuristics on C/N/T benchmark sets."
    )
    parser.add_argument("inputs", nargs="*", type=Path)
    args = parser.parse_args(argv)
    paths = sorted(args.inputs)
    if not paths:
        raise SystemExit("no input instances found")

    gaps: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        instance = load_instance(path)
        optimum = instance.reference_height
        category = _category(instance.name)
        for algo_name, pack in ALGORITHMS.items():
            _placements, height = pack(list(instance.items), instance.strip_width)
            gaps[category][algo_name].append(
                100 * (height - optimum) / optimum
            )

    header = "cat  " + "  ".join(f"{name:>6}" for name in ALGORITHMS)
    print(header)
    print("-" * len(header))
    for category in sorted(gaps):
        row = f"{category:<4}"
        for algo_name in ALGORITHMS:
            ours = sum(gaps[category][algo_name]) / len(gaps[category][algo_name])
            paper = PAPER_TABLE_2.get(category, {}).get(algo_name)
            row += f"  {ours:>3.1f}/{paper if paper is not None else '-'}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
