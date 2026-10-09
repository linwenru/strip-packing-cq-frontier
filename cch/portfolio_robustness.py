"""Portfolio robustness analyses on the 864-configuration gap matrix.

Rebuilds the full factorial matrix (216 base configs x {plain, tower removal,
vertical niche, both}) from the stage-1/2 run CSVs, verifies that greedy
forward selection reproduces the published stage-4 learning curve, then runs
two robustness checks:

1. random-portfolio control: distribution of 1000 random 12-config portfolios
   versus the greedy portfolio (3.80%);
2. leave-one-family-out (LOFO): portfolio re-selected on two instance
   families, evaluated on the held-out third.

Usage: python3 -m cch.portfolio_robustness
"""

from __future__ import annotations

import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    ("stage1-runs.csv", "twF+vnF"),
    ("stage2-tower-runs.csv", "twT+vnF"),
    ("stage2-vn-runs.csv", "twF+vnT"),
    ("stage2-vntower-runs.csv", "twT+vnT"),
]
FAMILIES = {"data/c": "C", "data/bkw": "BKW", "data/nt": "NT"}
K = 12


def load_matrix() -> tuple[dict[str, dict[str, float]], list[str], dict[str, str]]:
    """gap[config][instance] = gap_pct; also instance order and family map."""
    gap: dict[str, dict[str, float]] = defaultdict(dict)
    family: dict[str, str] = {}
    for filename, variant in VARIANTS:
        with open(ROOT / "docs" / filename, newline="") as handle:
            for row in csv.DictReader(handle):
                config = (
                    f"{row['ordering']}/{row['selection']}/{row['placement']}"
                    f"+{variant}"
                )
                gap[config][row["instance"]] = float(row["gap_pct"])
                family[row["instance"]] = FAMILIES[row["data_set"]]
    instances = sorted({i for g in gap.values() for i in g})
    return gap, instances, family


def portfolio_score(gap: dict[str, dict[str, float]], configs: list[str], instances: list[str]) -> float:
    total = 0.0
    for inst in instances:
        total += min(gap[c][inst] for c in configs)
    return total / len(instances)


def greedy_select(gap: dict[str, dict[str, float]], instances: list[str], k: int = K) -> tuple[list[str], list[float]]:
    configs = sorted(gap)
    chosen: list[str] = []
    curve: list[float] = []
    best = {inst: float("inf") for inst in instances}
    for _ in range(k):
        def score(config: str) -> float:
            return sum(min(best[i], gap[config][i]) for i in instances) / len(instances)
        winner = min(configs, key=lambda c: (score(c), c))
        chosen.append(winner)
        for inst in instances:
            best[inst] = min(best[inst], gap[winner][inst])
        curve.append(sum(best.values()) / len(instances))
    return chosen, curve


def main() -> None:
    gap, instances, family = load_matrix()
    print(f"matrix: {len(gap)} configs x {len(instances)} instances")

    # --- self-check: reproduce the stage-4 learning curve
    chosen, curve = greedy_select(gap, instances)
    print("\n[self-check] greedy curve vs the stage4 portfolio file:")
    with open(ROOT / "docs" / "stage4-portfolio.csv", newline="") as handle:
        published = [float(r["mean_gap_pct"]) for r in csv.DictReader(handle)]
    for k, (mine, pub) in enumerate(zip(curve, published), 1):
        flag = "OK" if abs(mine - pub) < 0.005 else "MISMATCH"
        print(f"  k={k:2d}  mine={mine:.4f}  published={pub:.4f}  {flag}")
    print("  chosen:", chosen[-1], "...", chosen[0])

    # --- random-portfolio control
    rng = random.Random(20260821)
    configs = sorted(gap)
    scores = []
    for _ in range(1000):
        sample = rng.sample(configs, K)
        scores.append(portfolio_score(gap, sample, instances))
    greedy = curve[-1]
    below = sum(1 for s in scores if s <= greedy)
    print(f"\n[random control] 1000 random 12-config portfolios: "
          f"mean={statistics.mean(scores):.3f}  sd={statistics.stdev(scores):.3f}  "
          f"min={min(scores):.3f}  best5%={sorted(scores)[49]:.3f}")
    print(f"  greedy 12-config = {greedy:.3f};  P(random <= greedy) = {below}/1000")

    # --- leave-one-family-out
    print("\n[LOFO] train on two families, test on the held-out third:")
    full_by_family = {}
    for fam in ("C", "BKW", "NT"):
        insts = [i for i in instances if family[i] == fam]
        full_by_family[fam] = portfolio_score(gap, chosen, insts)
    for fam in ("C", "BKW", "NT"):
        train = [i for i in instances if family[i] != fam]
        test = [i for i in instances if family[i] == fam]
        fold_configs, _ = greedy_select(gap, train)
        held = portfolio_score(gap, fold_configs, test)
        print(f"  held-out {fam:4s} (n={len(test):2d}): LOFO portfolio {held:.3f}  "
              f"vs full-data portfolio {full_by_family[fam]:.3f}")


if __name__ == "__main__":
    main()
