"""Command-line runner for the ISH (Wei et al. 2017) reproduction experiment.

Runs RandomLS with several seeds per instance and compares best/avg gap
against the paper's Appendix A Tables 5-6 (ISH columns). BKW01-13 correspond
to the paper's N1-N13.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from burke_bf import load_instance
from burke_bf.render import render_svg
from burke_bf.solver import validate_solution

from .model import Solution
from .rls import random_ls

# Paper Tables 5-6 (ISH column): instance -> (avg.gap, best.gap) in %.
PAPER_ISH = {
    "C1P1": (0, 0), "C1P2": (0, 0), "C1P3": (0, 0),
    "C2P1": (0, 0), "C2P2": (0, 0), "C2P3": (0, 0),
    "C3P1": (0, 0), "C3P2": (3.3, 0), "C3P3": (2, 0),
    "C4P1": (0.7, 0), "C4P2": (1.7, 1.7), "C4P3": (0, 0),
    "C5P1": (0.2, 0), "C5P2": (0, 0), "C5P3": (1.0, 0),
    "C6P1": (0.8, 0.8), "C6P2": (0.6, 0), "C6P3": (0.8, 0.8),
    "C7P1": (0.4, 0.4), "C7P2": (0.4, 0.4), "C7P3": (0.4, 0.4),
    "BKW01": (0, 0), "BKW02": (0, 0), "BKW03": (2, 0),
    "BKW04": (0, 0), "BKW05": (0, 0), "BKW06": (0.1, 0),
    "BKW07": (0, 0), "BKW08": (1.3, 1.3), "BKW09": (0.2, 0),
    "BKW10": (0, 0), "BKW11": (0, 0), "BKW12": (0.3, 0.3),
    "BKW13": (0.1, 0.1),
}


def _skyline_array(placements, strip_width: int) -> tuple[int, ...]:
    heights = [0] * strip_width
    for placement in placements:
        for x in range(placement.x, placement.right):
            heights[x] = max(heights[x], placement.top)
    return tuple(heights)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Wei et al.'s improved skyline heuristic (ISH)."
    )
    parser.add_argument(
        "inputs", nargs="*", type=Path, help="2DPackLib .ins2D files"
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="RandomLS runs per instance (seeds 1..N)"
    )
    parser.add_argument(
        "--time-limit", type=float, default=60.0,
        help="per-run time limit in seconds (default 60, as in the paper)",
    )
    parser.add_argument("--json", type=Path, help="write complete placements as JSON")
    parser.add_argument(
        "--svg-dir", type=Path, help="write an SVG of the best packing for every instance"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = sorted(args.inputs)
    if not paths:
        raise SystemExit("no input instances found")

    experiment: dict[str, object] = {
        "algorithm": "Wei et al. ISH (BestFitPack + RandomLS)",
        "runs": args.runs,
        "time_limit_s": args.time_limit,
        "instances": {},
    }
    print("instance  n    H    LB   ourBG  pBG   ourAG  pAG   time_s")
    print("-" * 62)
    for path in paths:
        instance = load_instance(path)
        lb = instance.reference_height
        best_height = None
        best_placements = None
        gaps = []
        elapsed = 0.0
        for seed in range(1, args.runs + 1):
            height, placements, t = random_ls(
                instance, time_limit=args.time_limit, seed=seed
            )
            elapsed += t
            if height is not None:
                if lb is not None:
                    gaps.append(100 * (height - lb) / lb)
                if best_height is None or height < best_height:
                    best_height, best_placements = height, placements

        our_best_gap = 100 * (best_height - lb) / lb if lb is not None and best_height is not None else None
        our_avg_gap = sum(gaps) / len(gaps) if gaps else None
        paper = PAPER_ISH.get(instance.name)

        solution = None
        if best_height is not None and best_placements is not None:
            solution = Solution(
                instance_name=instance.name,
                label="ish",
                height=best_height,
                placements=tuple(best_placements),
                skyline=_skyline_array(best_placements, instance.strip_width),
            )
            validate_solution(instance, solution)

        print(
            f"{instance.name:<8} {len(instance.items):>4}  "
            f"{str(best_height) if best_height is not None else '-':>4}  "
            f"{str(lb) if lb is not None else '-':>4}  "
            f"{f'{our_best_gap:.1f}' if our_best_gap is not None else '-':>5}  "
            f"{str(paper[1]) if paper is not None else '-':>5}  "
            f"{f'{our_avg_gap:.1f}' if our_avg_gap is not None else '-':>5}  "
            f"{str(paper[0]) if paper is not None else '-':>5}  "
            f"{elapsed:>7.1f}"
        )

        experiment["instances"][instance.name] = {
            "items": len(instance.items),
            "strip_width": instance.strip_width,
            "reference_height": instance.reference_height,
            "paper_ish_gaps": paper,
            "best_height": best_height,
            "best_gap": our_best_gap,
            "avg_gap": our_avg_gap,
            "elapsed_s": elapsed,
            "placements": (
                [
                    {
                        "item_id": p.item_id,
                        "x": p.x,
                        "y": p.y,
                        "width": p.width,
                        "height": p.height,
                        "rotated": p.rotated,
                    }
                    for p in best_placements
                ]
                if best_placements is not None
                else None
            ),
        }

        if args.svg_dir is not None and solution is not None:
            args.svg_dir.mkdir(parents=True, exist_ok=True)
            target = args.svg_dir / f"{instance.name.lower()}-ish.svg"
            target.write_text(render_svg(instance, solution), encoding="utf-8")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
