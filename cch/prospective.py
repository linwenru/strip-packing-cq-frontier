"""Prospective validation of component predictions on new distributions.

Pre-registered experiment (plan included in the data package, fixed
2026-09-11 before any run). Generates 120 new instances (four geometric
distribution groups x two sizes x 15 replications), evaluates a 144-config
grid (width ordering x 6 selections x 6 placements x tower-removal{+-} x
vertical-niche{+-}) plus two pre-fixed 12-config portfolios (P_mix from
the stage4 portfolio file, P_noVN re-selected by the same greedy rule from
the no-VN pool), and aggregates three tests:

- P1: TN placement reduces buried waste vs LM (paired within instance);
- P2: tower removal's gain comes mainly from top-waste reduction;
- P3: VN members' portfolio contribution on new distributions (open
  generalisation hypothesis, not directional).

Subcommands: generate | select-novn | run [--workers N] | aggregate
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

from burke_bf import load_instance
from burke_bf.solver import validate_solution
from fh import fast_heuristic

from .experiment import _lower_bound
from .mechanisms import measure
from .model import Config, Ordering, PlacementPolicy, Selection
from .portfolio_robustness import greedy_select
from .solver import solve_config

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "prospective"
RUNS_CSV = ROOT / "docs" / "prospective-runs.csv"
INSTANCES_CSV = ROOT / "docs" / "prospective-instances.csv"
SUMMARY_CSV = ROOT / "docs" / "prospective-summary.csv"
NOVN_CSV = ROOT / "docs" / "prospective-novn-portfolio.csv"
NOVN_TRL_CSV = ROOT / "docs" / "prospective-novn-trl-portfolio.csv"
MIX_CSV = ROOT / "docs" / "stage4-portfolio.csv"

GROUPS = ("a", "b", "c", "d")
SIZES = (100, 300)
N_INSTANCES = 15
STRIP_WIDTH = 1000
MASTER_SEED = 20260911
K = 12

HEADER = [
    "instance", "group", "n_items", "arm", "ordering", "selection",
    "placement", "tower_removal", "vertical_niche", "engine", "height",
    "lower_bound", "gap_pct", "raises", "buried_pct", "top_waste_pct",
    "time_ms",
]

# Two-sided 95% t critical values for the scopes in use (df = n - 1).
T_CRIT = {29: 2.0452, 119: 1.9801}


# ---------------------------------------------------------------- generation

def _seed(group_index: int, size_index: int, index: int) -> int:
    return MASTER_SEED + group_index * 10000 + size_index * 100 + index


def _gen_items(rng: random.Random, group: str, n: int) -> list[tuple[int, int, int]]:
    slender = group in ("b", "d")
    mixed = group in ("c", "d")
    items = []
    for j in range(1, n + 1):
        s = rng.uniform(80.0, 120.0)
        if mixed and rng.random() < 0.3:
            w = round(rng.uniform(0.15 * s, 0.35 * s))
            h = round(rng.uniform(0.15 * s, 0.35 * s))
        elif slender:
            w = round(rng.uniform(2.0 * s, 4.0 * s))
            h = round(rng.uniform(0.3 * s, 0.6 * s))
        else:
            w = round(rng.uniform(0.7 * s, 1.3 * s))
            h = round(rng.uniform(0.7 * s, 1.3 * s))
        items.append((j, max(w, 1), max(h, 1)))
    return items


def cmd_generate() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for g, group in enumerate(GROUPS):
        for s_idx, n in enumerate(SIZES):
            for i in range(1, N_INSTANCES + 1):
                rng = random.Random(_seed(g, s_idx, i))
                items = _gen_items(rng, group, n)
                path = DATA_DIR / f"pv-{group}-{n}-{i:02d}.ins2D"
                with open(path, "w") as handle:
                    handle.write(
                        f"# prospective group={group} n={n} idx={i}"
                        f" seed={_seed(g, s_idx, i)}\n"
                    )
                    handle.write(f"{n}\n{STRIP_WIDTH} -1\n")
                    for j, w, h in items:
                        handle.write(f"{j} {w} {h} 1 1 0\n")
                instance = load_instance(str(path))
                aspects = [
                    max(w, h) / min(w, h) for _, w, h in items
                ]
                rows.append(
                    {
                        "instance": instance.name,
                        "group": group,
                        "n_items": n,
                        "total_area": instance.total_area,
                        "mean_aspect": f"{statistics.mean(aspects):.3f}",
                        "lower_bound": _lower_bound(instance),
                    }
                )
    with open(INSTANCES_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} instances -> {DATA_DIR}")
    for group in GROUPS:
        for n in SIZES:
            sub = [r for r in rows if r["group"] == group and r["n_items"] == n]
            print(
                f"  {group} n={n:3d}: mean LB="
                f"{statistics.mean(int(r['lower_bound']) for r in sub):8.1f}"
                f"  mean aspect="
                f"{statistics.mean(float(r['mean_aspect']) for r in sub):.3f}"
            )


# ------------------------------------------------------- P_noVN portfolio

def _novn_pool(tr_locked: bool = False) -> dict[str, dict[str, float]]:
    """No-VN pool: 432 configs (216 base x 2 TR states); with ``tr_locked``
    only the 216 tower-removal-on configs (sensitivity variant isolating
    the VN contrast from TR composition)."""
    variants = (
        ("stage2-tower-runs.csv", "twT+vnF"),
    ) if tr_locked else (
        ("stage1-runs.csv", "twF+vnF"),
        ("stage2-tower-runs.csv", "twT+vnF"),
    )
    gap: dict[str, dict[str, float]] = {}
    for filename, variant in variants:
        with open(ROOT / "docs" / filename, newline="") as handle:
            for row in csv.DictReader(handle):
                config = (
                    f"{row['ordering']}/{row['selection']}/"
                    f"{row['placement']}+{variant}"
                )
                gap.setdefault(config, {})[row["instance"]] = float(
                    row["gap_pct"]
                )
    return gap


def cmd_select_novn(tr_locked: bool = False) -> None:
    gap = _novn_pool(tr_locked)
    out_csv = NOVN_TRL_CSV if tr_locked else NOVN_CSV
    instances = sorted({i for g in gap.values() for i in g})
    chosen, curve = greedy_select(gap, instances, k=K)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "mean_gap_pct", "added_config"])
        for k, (config, score) in enumerate(zip(chosen, curve), 1):
            writer.writerow([k, f"{score:.4f}", config])
    print(f"no-VN pool: {len(gap)} configs x {len(instances)} instances"
          f"{' (TR-locked)' if tr_locked else ''}")
    print(f"greedy {K}-config no-VN portfolio -> {out_csv}")
    for k, (config, score) in enumerate(zip(chosen, curve), 1):
        print(f"  k={k:2d}  {score:.4f}  {config}")


def _parse_config(text: str) -> Config:
    base, *flags = text.split("+")
    ordering, selection, placement = base.split("/")
    return Config(
        ordering=Ordering(ordering),
        selection=Selection(selection),
        placement=PlacementPolicy(placement),
        tower_removal="twT" in flags,
        vertical_niche="vnT" in flags,
    )


def _portfolio_configs(path: Path) -> list[Config]:
    with open(path, newline="") as handle:
        return [_parse_config(row["added_config"]) for row in csv.DictReader(handle)]


# --------------------------------------------------------------------- run

def _grid_configs() -> list[Config]:
    return [
        Config(
            ordering=Ordering.WIDTH,
            selection=selection,
            placement=placement,
            tower_removal=tower_removal,
            vertical_niche=vertical_niche,
        )
        for selection in Selection
        for placement in PlacementPolicy
        for tower_removal in (False, True)
        for vertical_niche in (False, True)
    ]


def _done_keys() -> set[tuple[str, ...]]:
    try:
        with open(RUNS_CSV, newline="") as handle:
            return {
                (
                    row["instance"], row["arm"], row["ordering"],
                    row["selection"], row["placement"], row["tower_removal"],
                    row["vertical_niche"],
                )
                for row in csv.DictReader(handle)
            }
    except FileNotFoundError:
        return set()


def _self_check(paths: list[Path]) -> None:
    """measure() and solve_config() must agree on VN-off configurations."""
    probes = {
        "pv-a-100-01": None,
        "pv-b-300-01": None,
        "pv-d-100-01": None,
    }
    mismatches = 0
    for path in paths:
        if path.stem not in probes:
            continue
        instance = load_instance(str(path))
        for selection in Selection:
            for placement in PlacementPolicy:
                for tower_removal in (False, True):
                    config = Config(
                        ordering=Ordering.WIDTH,
                        selection=selection,
                        placement=placement,
                        tower_removal=tower_removal,
                    )
                    h_measure = max(
                        measure(instance, config, tower_removal=tower_removal)[5]
                    )
                    h_solve = max(solve_config(instance, config)[1])
                    if h_measure != h_solve:
                        print(
                            f"MISMATCH {path.stem} {config.value}: "
                            f"measure={h_measure} solve={h_solve}"
                        )
                        mismatches += 1
    if mismatches:
        raise SystemExit(f"self-check failed: {mismatches} mismatches")
    print("self-check OK: measure == solve_config (72 configs x 3 instances)")


def _work(task: tuple) -> list[str]:
    path_s, group, n_items, arm, config, validate = task
    path = Path(path_s)
    instance = load_instance(path_s)
    lower_bound = _lower_bound(instance)
    strip_area = instance.strip_width
    start = time.perf_counter()
    raises = buried_pct = top_pct = ""
    if arm == "fh":
        # FH background baseline (Leung & Zhang 2011), same swap-budget
        # convention as cch.baselines: full pass for n <= 250, else 5000.
        budget = None if len(instance.items) <= 250 else 5000
        placements, height = fast_heuristic(
            list(instance.items), instance.strip_width, swap_budget=budget
        )
        if validate:
            validate_solution(
                instance,
                SimpleNamespace(placements=tuple(placements), height=height),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return [
            instance.name, group, str(n_items), arm, "", "", "", "", "",
            "fh", str(height), str(lower_bound),
            f"{100.0 * (height - lower_bound) / lower_bound:.4f}",
            "", "", "", f"{elapsed_ms:.1f}",
        ]
    if arm != "grid" or config.vertical_niche:
        engine = "solve"
        placements, skyline, _ = solve_config(instance, config)
        height = max(skyline)
        if validate:
            validate_solution(
                instance,
                SimpleNamespace(placements=tuple(placements), height=height),
            )
    else:
        engine = "measure"
        n_raises, buried, _exact, _placed, _res, skyline = measure(
            instance, config, tower_removal=config.tower_removal
        )
        height = max(skyline)
        raises = str(n_raises)
        buried_pct = f"{100.0 * buried / (lower_bound * strip_area):.4f}"
        top = height * strip_area - instance.total_area - buried
        top_pct = f"{100.0 * top / (lower_bound * strip_area):.4f}"
        if validate:
            placements, skyline2, _ = solve_config(instance, config)
            if max(skyline2) != height:
                raise RuntimeError(
                    f"engine mismatch {path.stem} {config.value}: "
                    f"measure={height} solve={max(skyline2)}"
                )
            validate_solution(
                instance,
                SimpleNamespace(placements=tuple(placements), height=height),
            )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return [
        instance.name,
        group,
        str(n_items),
        arm,
        config.ordering.value,
        config.selection.value,
        config.placement.value,
        "T" if config.tower_removal else "F",
        "T" if config.vertical_niche else "F",
        engine,
        str(height),
        str(lower_bound),
        f"{100.0 * (height - lower_bound) / lower_bound:.4f}",
        raises,
        buried_pct,
        top_pct,
        f"{elapsed_ms:.1f}",
    ]


def cmd_run(workers: int) -> None:
    paths = sorted(DATA_DIR.glob("pv-*.ins2D"))
    if len(paths) != len(GROUPS) * len(SIZES) * N_INSTANCES:
        raise SystemExit(
            f"expected {len(GROUPS) * len(SIZES) * N_INSTANCES} instances in "
            f"{DATA_DIR}, found {len(paths)} -- run `generate` first"
        )
    portfolios = {"pmix": _portfolio_configs(MIX_CSV)}
    if NOVN_CSV.exists():
        portfolios["ponovn"] = _portfolio_configs(NOVN_CSV)
    if NOVN_TRL_CSV.exists():
        portfolios["ponovn-trl"] = _portfolio_configs(NOVN_TRL_CSV)
    grid = _grid_configs()
    done = _done_keys()
    tasks = []
    for path in paths:
        _pv, group, n_items, _idx = path.stem.split("-")
        validate = path.stem.endswith("-01")
        fh_key = (path.stem.replace("-", "").upper(), "fh", "", "", "", "", "")
        if fh_key not in done:
            tasks.append((str(path), group, n_items, "fh", None, validate))
        for arm, configs in (("grid", grid), *portfolios.items()):
            for config in configs:
                key = (
                    path.stem.replace("-", "").upper(),
                    arm,
                    config.ordering.value,
                    config.selection.value,
                    config.placement.value,
                    "T" if config.tower_removal else "F",
                    "T" if config.vertical_niche else "F",
                )
                if key in done:
                    continue
                tasks.append(
                    (str(path), group, n_items, arm, config, validate)
                )
    print(f"{len(tasks)} runs to do ({len(done)} done)")
    if not tasks:
        return
    _self_check(paths)
    fresh = not done
    completed = 0
    start = time.perf_counter()
    with open(RUNS_CSV, "a", newline="") as handle:
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(HEADER)
        with Pool(workers) as pool:
            for row in pool.imap_unordered(_work, tasks, chunksize=8):
                writer.writerow(row)
                completed += 1
                if completed % 500 == 0:
                    handle.flush()
                    rate = completed / (time.perf_counter() - start)
                    print(
                        f"  {completed}/{len(tasks)} "
                        f"({rate:.0f} runs/s)", flush=True
                    )
    print(f"done: {completed} runs in {time.perf_counter() - start:.0f}s")


# --------------------------------------------------------------- statistics

def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betainc(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_p_two_sided(t: float, df: int) -> float:
    return _betainc(df / (df + t * t), df / 2.0, 0.5)


def _bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        running = min(running, pvals[i] * m / rank)
        adj[i] = running
    return adj


def _summarise(values: list[float]) -> dict[str, float]:
    n = len(values)
    mu = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mu / se if se > 0 else 0.0
    df = n - 1
    crit = T_CRIT.get(df, 1.96)
    return {
        "n": n,
        "mean": mu,
        "sd": sd,
        "ci_lo": mu - crit * se,
        "ci_hi": mu + crit * se,
        "t": t,
        "p": _t_p_two_sided(abs(t), df) if se > 0 else 1.0,
    }


# --------------------------------------------------------------- aggregate

def _index_grid(rows: list[dict[str, str]]) -> dict[tuple, dict[str, str]]:
    return {
        (
            r["instance"], r["selection"], r["placement"],
            r["tower_removal"], r["vertical_niche"],
        ): r
        for r in rows
        if r["arm"] == "grid"
    }


def cmd_aggregate() -> None:
    with open(RUNS_CSV, newline="") as handle:
        rows = list(csv.DictReader(handle))
    grid = _index_grid(rows)
    groups_of = {r["instance"]: r["group"] for r in rows}
    instances = sorted(groups_of)
    selections = sorted({r["selection"] for r in rows if r["arm"] == "grid"})
    placements = [p.value for p in PlacementPolicy]

    scopes = [(g, [i for i in instances if groups_of[i] == g]) for g in GROUPS]
    scopes.append(("all", instances))

    # --- P1: TN vs LM, buried waste and gap (TR off, VN off)
    p1_buried: dict[str, float] = {}
    p1_gap: dict[str, float] = {}
    for inst in instances:
        diffs_b, diffs_g = [], []
        for sel in selections:
            tn = grid[(inst, sel, "TN", "F", "F")]
            lm = grid[(inst, sel, "LM", "F", "F")]
            diffs_b.append(float(lm["buried_pct"]) - float(tn["buried_pct"]))
            diffs_g.append(float(lm["gap_pct"]) - float(tn["gap_pct"]))
        p1_buried[inst] = statistics.mean(diffs_b)
        p1_gap[inst] = statistics.mean(diffs_g)

    # --- P2: tower removal on/off, fitness-number, six placements (VN off)
    p2_top: dict[str, float] = {}
    p2_buried: dict[str, float] = {}
    p2_gap: dict[str, float] = {}
    for inst in instances:
        d_top, d_bur, d_gap = [], [], []
        for pla in placements:
            off = grid[(inst, "fitness-number", pla, "F", "F")]
            on = grid[(inst, "fitness-number", pla, "T", "F")]
            d_top.append(float(off["top_waste_pct"]) - float(on["top_waste_pct"]))
            d_bur.append(float(off["buried_pct"]) - float(on["buried_pct"]))
            d_gap.append(float(off["gap_pct"]) - float(on["gap_pct"]))
        p2_top[inst] = statistics.mean(d_top)
        p2_buried[inst] = statistics.mean(d_bur)
        p2_gap[inst] = statistics.mean(d_gap)

    # --- P3: portfolio per-instance best height
    arm_names = ["pmix", "ponovn"]
    if any(r["arm"] == "ponovn-trl" for r in rows):
        arm_names.append("ponovn-trl")
    best: dict[str, dict[str, tuple[int, int, float]]] = {}
    for arm in arm_names:
        per_inst: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            if r["arm"] == arm:
                per_inst.setdefault(r["instance"], []).append(r)
        best[arm] = {
            inst: (
                min(int(r["height"]) for r in sub),
                int(sub[0]["lower_bound"]),
                sum(float(r["time_ms"]) for r in sub),
            )
            for inst, sub in per_inst.items()
        }

    def _portfolio_diff(arm: str) -> dict[str, float]:
        return {
            inst: 100.0 * (best[arm][inst][0] - best["pmix"][inst][0])
            / best["pmix"][inst][1]
            for inst in instances
        }

    p3_diff = _portfolio_diff("ponovn")
    p3b_diff = (
        _portfolio_diff("ponovn-trl") if "ponovn-trl" in best else None
    )

    # --- report
    endpoints = [
        ("P1", "buried% LM-TN", p1_buried),
        ("P1", "gap% LM-TN", p1_gap),
        ("P2", "top% TRoff-TRon", p2_top),
        ("P2", "buried% TRoff-TRon", p2_buried),
        ("P2", "gap% TRoff-TRon", p2_gap),
        ("P3", "bestgap% noVN-mix", p3_diff),
    ]
    if p3b_diff is not None:
        endpoints.append(("P3b", "bestgap% noVNtrl-mix", p3b_diff))
    out_rows = []
    print(f"{'test':4s} {'endpoint':22s} {'scope':6s} {'n':>4s} {'mean':>8s} "
          f"{'sd':>7s} {'95% CI':>18s} {'t':>7s} {'p':>9s}")
    for test, endpoint, values in endpoints:
        for scope, insts in scopes:
            stats = _summarise([values[i] for i in insts])
            out_rows.append(
                {
                    "test": test,
                    "endpoint": endpoint,
                    "scope": scope,
                    **{k: (f"{v:.4f}" if isinstance(v, float) else v)
                       for k, v in stats.items()},
                }
            )
            if scope == "all" or test in ("P1", "P2", "P3", "P3b"):
                print(
                    f"{test:4s} {endpoint:22s} {scope:6s} {stats['n']:4d} "
                    f"{stats['mean']:+8.4f} {stats['sd']:7.4f} "
                    f"[{stats['ci_lo']:+7.4f},{stats['ci_hi']:+7.4f}] "
                    f"{stats['t']:+7.2f} {stats['p']:9.2e}"
                )

    primary = [
        _summarise([p1_buried[i] for i in instances])["p"],
        _summarise([p2_top[i] for i in instances])["p"],
        _summarise([p3_diff[i] for i in instances])["p"],
    ]
    adj = _bh_adjust(primary)
    print("\nBH-adjusted primary p-values (overall):")
    for name, p, q in zip(("P1", "P2", "P3"), primary, adj):
        print(f"  {name}: p={p:.3e}  p_BH={q:.3e}")

    # P2 sanity: identity gap == top + buried in normalised units
    ident = [
        p2_gap[i] - (p2_top[i] + p2_buried[i]) for i in instances
    ]
    max_dev = max(abs(v) for v in ident)
    print(f"\nP2 identity check: max |gap-(top+buried)| = {max_dev:.6f}")

    # P3 extras: win/tie/loss and runtime ratio
    def _wtool(diffs: dict[str, float]) -> tuple[int, int, int]:
        return (
            sum(1 for i in instances if diffs[i] < -1e-9),
            sum(1 for i in instances if abs(diffs[i]) <= 1e-9),
            sum(1 for i in instances if diffs[i] > 1e-9),
        )

    wins, ties, losses = _wtool(p3_diff)
    t_mix = statistics.mean(best["pmix"][i][2] for i in instances)
    t_no = statistics.mean(best["ponovn"][i][2] for i in instances)
    print(f"P3 win/tie/loss (noVN-mix <0 = mix better): "
          f"{wins}/{ties}/{losses}")
    print(f"P3 mean total time per instance: P_mix={t_mix:.0f}ms "
          f"P_noVN={t_no:.0f}ms ratio={t_no / t_mix:.2f}")
    if p3b_diff is not None:
        wins, ties, losses = _wtool(p3b_diff)
        t_trl = statistics.mean(best["ponovn-trl"][i][2] for i in instances)
        print(f"P3b (TR-locked sensitivity) win/tie/loss: "
              f"{wins}/{ties}/{losses}")
        print(f"P3b mean total time per instance: P_noVN-TRL={t_trl:.0f}ms "
              f"ratio vs mix={t_trl / t_mix:.2f}")

    # context: grid difficulty per group, plus the FH background baseline
    print("\ncontext: mean grid gap% by group (all 144 configs):")
    fh = {r["instance"]: float(r["gap_pct"]) for r in rows if r["arm"] == "fh"}
    for scope, insts in scopes:
        gaps = [
            float(r["gap_pct"])
            for r in rows
            if r["arm"] == "grid" and r["instance"] in set(insts)
        ]
        extra = (
            f"  FH={statistics.mean(fh[i] for i in insts):7.3f}"
            if fh else ""
        )
        print(f"  {scope:6s} {statistics.mean(gaps):7.3f}{extra}")

    with open(SUMMARY_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nwrote {SUMMARY_CSV}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    sel_p = sub.add_parser("select-novn")
    sel_p.add_argument("--tr-locked", action="store_true",
                       help="restrict the pool to tower-removal-on configs "
                            "(216; sensitivity variant)")
    run_p = sub.add_parser("run")
    run_p.add_argument("--workers", type=int, default=6)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate()
    elif args.command == "select-novn":
        cmd_select_novn(tr_locked=args.tr_locked)
    elif args.command == "run":
        cmd_run(args.workers)
    elif args.command == "aggregate":
        cmd_aggregate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
