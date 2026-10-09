"""Command-line runner for the Wei et al. skyline heuristic reproduction.

Runs IDBS (rotatable variant) per instance and compares against the optimal
height (the paper's IDBS found the optimum on every C/Burke/Babu instance,
Table 2). BKW01-13 correspond to the paper's Burke test set N1-N13.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from burke_bf import load_instance
from burke_bf.render import render_svg
from burke_bf.solver import validate_solution

from .idbs import idbs
from .model import Solution
from .solver import skyline_array

# Paper Table 2 (rotatable variant), IDBS columns: relative gap in %.
PAPER_TABLE_2 = {
    "C": (0.0, 0.0),
    "Burke": (0.0, 0.0),
    "Babu": (0.0, 0.0),
    "N": (0.80, 1.03),
    "T": (0.80, 1.20),
    "CX": (0.19, 0.26),
    "Nice": (0.92, 1.27),
    "Path": (1.17, 1.43),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Wei et al.'s skyline heuristic via IDBS."
    )
    parser.add_argument(
        "inputs", nargs="*", type=Path, help="2DPackLib .ins2D files"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=100.0,
        help="per-instance IDBS time limit in seconds (default 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="tabu search random seed"
    )
    parser.add_argument(
        "--no-tabu",
        action="store_true",
        help="disable the tabu search (paper's 'No Tabu' variant)",
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
        "algorithm": "Wei et al. skyline heuristic IDBS",
        "rotatable": True,
        "tabu": not args.no_tabu,
        "seed": args.seed,
        "instances": {},
    }
    label = "idbs" if not args.no_tabu else "no-tabu"
    print("instance  n    H    optimum  match  time_s")
    print("-" * 44)
    matches = compared = 0
    for path in paths:
        instance = load_instance(path)
        started = perf_counter()
        height, placements, elapsed = idbs(
            instance,
            time_limit=args.time_limit,
            seed=args.seed,
            use_tabu=not args.no_tabu,
        )
        optimum = instance.reference_height
        match = ""
        solution = None
        if height is not None and placements is not None:
            solution = Solution(
                instance_name=instance.name,
                label=label,
                height=height,
                placements=tuple(placements),
                skyline=skyline_array(placements, instance.strip_width),
            )
            validate_solution(instance, solution)
        if optimum is not None:
            compared += 1
            match = "yes" if height == optimum else "NO"
            matches += height == optimum
        print(
            f"{instance.name:<8} {len(instance.items):>4}  "
            f"{str(height) if height is not None else '-':>4}  "
            f"{str(optimum) if optimum is not None else '-':>7}  "
            f"{match:>5}  {elapsed:>7.2f}"
        )

        experiment["instances"][instance.name] = {
            "items": len(instance.items),
            "strip_width": instance.strip_width,
            "reference_height": instance.reference_height,
            "height": height,
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
                    for p in placements
                ]
                if placements is not None
                else None
            ),
        }

        if args.svg_dir is not None and solution is not None:
            args.svg_dir.mkdir(parents=True, exist_ok=True)
            target = args.svg_dir / f"{instance.name.lower()}-{label}.svg"
            target.write_text(render_svg(instance, solution), encoding="utf-8")

    print("-" * 44)
    print(f"reached optimum on {matches}/{compared} instances")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
