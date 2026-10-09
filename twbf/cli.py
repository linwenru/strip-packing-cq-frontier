"""Command-line runner for the TWBF reproduction experiment.

Compares computed heights against Table A.3 of Appendix A of Verstichel's
PhD thesis (the open-access summary of Verstichel et al. 2013), which lists
their own best-fit implementation, the three-way best-fit heuristic and the
optimal time three-way heuristic. BKW01-13 correspond to the paper's N1-N13.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from burke_bf import load_instance, solve
from burke_bf.render import render_svg

from .model import Solution
from .solver import solve as solve_tw

# Table A.3: instance -> (their BF, three-way BF, optimal time three-way).
PAPER_TABLE_A3 = {
    "BKW01": (45, 44, 45),
    "BKW02": (53, 53, 53),
    "BKW03": (54, 52, 54),
    "BKW04": (86, 86, 86),
    "BKW05": (105, 104, 104),
    "BKW06": (102, 102, 102),
    "BKW07": (107, 106, 107),
    "BKW08": (83, 83, 83),
    "BKW09": (163, 154, 163),
    "BKW10": (153, 152, 152),
    "BKW11": (153, 152, 152),
    "BKW12": (305, 305, 305),
    "BKW13": (964, 964, 964),
    "C1P1": (21, 21, 21),
    "C1P2": (22, 21, 21),
    "C1P3": (24, 24, 24),
    "C2P1": (16, 16, 16),
    "C2P2": (16, 16, 16),
    "C2P3": (16, 16, 16),
    "C3P1": (32, 32, 32),
    "C3P2": (34, 32, 34),
    "C3P3": (33, 32, 33),
    "C4P1": (63, 63, 63),
    "C4P2": (62, 62, 62),
    "C4P3": (62, 62, 62),
    "C5P1": (93, 92, 92),
    "C5P2": (92, 92, 92),
    "C5P3": (93, 93, 93),
    "C6P1": (123, 123, 123),
    "C6P2": (122, 122, 122),
    "C6P3": (124, 123, 123),
    "C7P1": (246, 244, 246),
    "C7P2": (244, 244, 244),
    "C7P3": (245, 245, 245),
}


def _solution_record(solution: Solution) -> dict[str, object]:
    return {
        "combination": solution.combination.value,
        "height": solution.height,
        "initial_height": solution.initial_height,
        "tower_moves": solution.tower_moves,
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
        description="Run Verstichel et al.'s three-way best-fit heuristic."
    )
    parser.add_argument(
        "inputs", nargs="*", type=Path, help="2DPackLib .ins2D files"
    )
    parser.add_argument(
        "--json", type=Path, help="write complete placements as JSON"
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
        "algorithm": "Verstichel et al. three-way best fit",
        "instances": {},
    }
    print("instance  n    ourBF  pBF  TWBF  pTWBF  pOT  match  time_s")
    print("-" * 60)
    matches = compared = 0
    for path in paths:
        instance = load_instance(path)
        started = perf_counter()
        best, _heights = solve_tw(instance)
        our_bf, _ = solve(instance)
        elapsed = perf_counter() - started
        paper = PAPER_TABLE_A3.get(instance.name)
        match = ""
        if paper is not None:
            compared += 1
            match = "yes" if best.height == paper[1] else "NO"
            matches += best.height == paper[1]
        print(
            f"{instance.name:<8} {len(instance.items):>4}  "
            f"{our_bf.height:>5}  "
            f"{str(paper[0]) if paper is not None else '-':>4}  "
            f"{best.height:>4}  "
            f"{str(paper[1]) if paper is not None else '-':>5}  "
            f"{str(paper[2]) if paper is not None else '-':>4}  "
            f"{match:>5}  {elapsed:>7.2f}"
        )

        experiment["instances"][instance.name] = {
            "items": len(instance.items),
            "strip_width": instance.strip_width,
            "reference_height": instance.reference_height,
            "paper_table_a3": paper,
            "our_bf_height": our_bf.height,
            "elapsed_s": elapsed,
            "best_combination": best.combination.value,
            "best": _solution_record(best),
        }

        if args.svg_dir is not None:
            args.svg_dir.mkdir(parents=True, exist_ok=True)
            target = args.svg_dir / f"{instance.name.lower()}-best.svg"
            target.write_text(render_svg(instance, best), encoding="utf-8")

    print("-" * 60)
    print(f"matched paper Table A.3 (TWBF) on {matches}/{compared} instances")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
