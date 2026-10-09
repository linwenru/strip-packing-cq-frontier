"""Command-line runner for the FH (Leung & Zhang 2011) reproduction.

Runs FH on the C (Hopper & Turton) and BKW (Burke N1-N13) sets and compares
against the paper's Table 1 and Table 2 (FH columns).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from burke_bf import load_instance, Solution, Policy
from burke_bf.solver import validate_solution

from .solver import fast_heuristic

# Paper Tables 1-2 (FH column): expected height per instance.
PAPER_FH = {
    "C1P1": 20, "C1P2": 20, "C1P3": 21,
    "C2P1": 16, "C2P2": 15, "C2P3": 15,
    "C3P1": 31, "C3P2": 31, "C3P3": 32,
    "C4P1": 61, "C4P2": 61, "C4P3": 61,
    "C5P1": 91, "C5P2": 90, "C5P3": 91,
    "C6P1": 121, "C6P2": 121, "C6P3": 121,
    "C7P1": 241, "C7P2": 241, "C7P3": 241,
    "BKW01": 40, "BKW02": 52, "BKW03": 51, "BKW04": 83,
    "BKW05": 102, "BKW06": 101, "BKW07": 102, "BKW08": 81,
    "BKW09": 151, "BKW10": 151, "BKW11": 151, "BKW12": 301,
    "BKW13": 960,
}


def _skyline_array(placements, strip_width: int) -> tuple[int, ...]:
    heights = [0] * strip_width
    for placement in placements:
        for x in range(placement.x, placement.right):
            heights[x] = max(heights[x], placement.top)
    return tuple(heights)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Leung & Zhang's FH.")
    parser.add_argument("inputs", nargs="*", type=Path)
    args = parser.parse_args(argv)
    paths = sorted(args.inputs)
    if not paths:
        raise SystemExit("no input instances found")

    print("instance  n    H    paper  match  time_s")
    print("-" * 44)
    matches = compared = 0
    for path in paths:
        instance = load_instance(path)
        started = perf_counter()
        placements, height = fast_heuristic(
            list(instance.items), instance.strip_width
        )
        elapsed = perf_counter() - started
        solution = Solution(
            instance_name=instance.name,
            policy=Policy.LEFTMOST,
            height=height,
            initial_height=height,
            placements=tuple(placements),
            skyline=_skyline_array(placements, instance.strip_width),
            tower_moves=0,
        )
        validate_solution(instance, solution)
        paper = PAPER_FH.get(instance.name)
        match = ""
        if paper is not None:
            compared += 1
            match = "yes" if height == paper else "NO"
            matches += height == paper
        print(
            f"{instance.name:<8} {len(instance.items):>4}  "
            f"{height:>4}  {str(paper) if paper is not None else '-':>6}  "
            f"{match:>5}  {elapsed:>7.2f}"
        )
    print("-" * 44)
    print(f"matched paper on {matches}/{compared} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
