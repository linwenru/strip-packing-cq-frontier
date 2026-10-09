"""Command-line runner for the BBF reproduction experiment.

Compares computed heights against the paper's Table 6 (Aşık & Özcan 2009),
which lists BBF and BF heights per instance. BKW01–BKW13 correspond to the
paper's N1–N13.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from burke_bf.io import load_instance
from burke_bf.render import render_svg

from .model import Combination, Solution
from .solver import solve

# Paper Table 6: instance -> (BBF height, BF height). BKW01-13 are N1-N13.
PAPER_TABLE_6 = {
    "M1": (9, 13),
    "BKW01": (40, 45),
    "BKW02": (52, 53),
    "BKW03": (52, 52),
    "BKW04": (82, 83),
    "BKW05": (104, 105),
    "BKW06": (102, 103),
    "BKW07": (106, 107),
    "BKW08": (82, 84),
    "BKW09": (152, 152),
    "BKW10": (151, 152),
    "BKW11": (151, 152),
    "BKW12": (303, 306),
    "BKW13": (964, 964),
    "C1P1": (20, 21),
    "C1P2": (21, 22),
    "C1P3": (21, 24),
    "C2P1": (16, 16),
    "C2P2": (16, 16),
    "C2P3": (15, 16),
    "C3P1": (30, 32),
    "C3P2": (33, 34),
    "C3P3": (31, 33),
    "C4P1": (62, 63),
    "C4P2": (62, 62),
    "C4P3": (61, 62),
    "C5P1": (91, 93),
    "C5P2": (92, 92),
    "C5P3": (91, 93),
    "C6P1": (122, 123),
    "C6P2": (121, 122),
    "C6P3": (122, 124),
    "C7P1": (243, 247),
    "C7P2": (244, 244),
    "C7P3": (244, 245),
}


def _solution_record(solution: Solution) -> dict[str, object]:
    return {
        "combination": solution.combination.value,
        "height": solution.height,
        "placements": [
            {
                "item_id": placement.item_id,
                "x": placement.x,
                "y": placement.y,
                "width": placement.width,
                "height": placement.height,
                "rotated": placement.rotated,
            }
            for placement in solution.placements
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Aşık and Özcan's bidirectional best-fit heuristic."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="2DPackLib .ins2D files",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="write complete placements and summary data as JSON",
    )
    parser.add_argument(
        "--svg-dir",
        type=Path,
        help="write an SVG of the best packing for every instance",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = sorted(args.inputs)
    if not paths:
        raise SystemExit("no input instances found")

    experiment: dict[str, object] = {
        "algorithm": "Asik-Ozcan bidirectional best fit",
        "instances": {},
    }
    print("instance  n    BBF  optimum  paperBBF  paperBF  gap  match  time_s")
    print("-" * 72)
    matches = 0
    compared = 0
    for path in paths:
        instance = load_instance(path)
        started = perf_counter()
        best, _heights = solve(instance)
        elapsed = perf_counter() - started
        optimum = instance.reference_height
        gap = best.height - optimum if optimum is not None else None
        paper = PAPER_TABLE_6.get(instance.name)
        match = ""
        if paper is not None:
            compared += 1
            match = "yes" if best.height == paper[0] else "NO"
            matches += best.height == paper[0]
        print(
            f"{instance.name:<8} {len(instance.items):>4}  "
            f"{best.height:>4}  "
            f"{str(optimum) if optimum is not None else '-':>7}  "
            f"{str(paper[0]) if paper is not None else '-':>8}  "
            f"{str(paper[1]) if paper is not None else '-':>7}  "
            f"{str(gap) if gap is not None else '-':>3}  "
            f"{match:>5}  {elapsed:>7.2f}"
        )

        experiment["instances"][instance.name] = {
            "items": len(instance.items),
            "strip_width": instance.strip_width,
            "reference_height": instance.reference_height,
            "paper_table_6": paper,
            "elapsed_s": elapsed,
            "best_combination": best.combination.value,
            "best": _solution_record(best),
        }

        if args.svg_dir is not None:
            args.svg_dir.mkdir(parents=True, exist_ok=True)
            target = args.svg_dir / f"{instance.name.lower()}-best.svg"
            target.write_text(render_svg(instance, best), encoding="utf-8")

    print("-" * 72)
    print(f"matched paper Table 6 on {matches}/{compared} instances")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
