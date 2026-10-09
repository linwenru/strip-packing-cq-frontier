"""Command-line runner for the C7 reproduction experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .io import load_instance
from .model import Policy, Solution
from .render import render_svg
from .solver import solve


PAPER_TABLE_3 = {
    "C7P1": {"LM": 246, "TN": 247, "SN": 250},
    "C7P2": {"LM": 246, "TN": 244, "SN": 246},
    "C7P3": {"LM": 245, "TN": 246, "SN": 248},
}
PAPER_TABLE_5 = {"C7P1": 247, "C7P2": 244, "C7P3": 245}


def _default_inputs() -> list[Path]:
    return sorted(
        Path("reproductions/burke_2004/data/c7").glob("*.ins2D")
    )


def _solution_record(solution: Solution) -> dict[str, object]:
    return {
        "policy": solution.policy.value,
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
        description=(
            "Run Burke, Kendall, and Whitwell's offline best-fit heuristic."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "2DPackLib .ins2D files "
            "(default: reproductions/burke_2004/data/c7/*.ins2D)"
        ),
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="disable the paper's tower-removal postprocessing stage",
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
    paths = args.inputs or _default_inputs()
    if not paths:
        raise SystemExit("no input instances found")

    experiment: dict[str, object] = {
        "algorithm": "Burke-Kendall-Whitwell best fit",
        "postprocess": not args.no_postprocess,
        "instances": {},
    }
    print(
        "instance  n    LM  TN  SN  best  optimum  table5  gap  time_ms"
    )
    print("-" * 68)
    for path in paths:
        instance = load_instance(path)
        started = perf_counter()
        best, by_policy = solve(
            instance, postprocess=not args.no_postprocess
        )
        elapsed_ms = (perf_counter() - started) * 1000
        optimum = instance.reference_height
        gap = best.height - optimum if optimum is not None else None
        heights = {
            policy.short_name: by_policy[policy].height for policy in Policy
        }
        print(
            f"{instance.name:<8} {len(instance.items):>3}  "
            f"{heights['LM']:>3} {heights['TN']:>3} {heights['SN']:>3}  "
            f"{best.height:>4}  "
            f"{str(optimum) if optimum is not None else '-':>7}  "
            f"{str(PAPER_TABLE_5.get(instance.name, '-')):>6}  "
            f"{str(gap) if gap is not None else '-':>3}  "
            f"{elapsed_ms:>7.2f}"
        )

        instance_record = {
            "items": len(instance.items),
            "strip_width": instance.strip_width,
            "reference_height": instance.reference_height,
            "paper_table_3": PAPER_TABLE_3.get(instance.name),
            "paper_table_5": PAPER_TABLE_5.get(instance.name),
            "elapsed_ms": elapsed_ms,
            "best_policy": best.policy.value,
            "best": _solution_record(best),
            "by_policy": {
                policy.value: _solution_record(solution)
                for policy, solution in by_policy.items()
            },
        }
        experiment["instances"][instance.name] = instance_record

        if args.svg_dir is not None:
            args.svg_dir.mkdir(parents=True, exist_ok=True)
            target = args.svg_dir / f"{instance.name.lower()}-best.svg"
            target.write_text(render_svg(instance, best), encoding="utf-8")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
