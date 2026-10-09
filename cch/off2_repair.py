"""OFF-2 validation C: defect-guided repair region selection (design-c.md).

Only the candidate-window height-limit precheck (v2 C.3) is implemented and
run at this stage — the frozen repair comparison follows after the precheck
results are reviewed:

    python3 -m cch.off2_repair precheck   # -> c-precheck-{instances,windows}.csv
    python3 -m cch.off2_repair run        # -> c-repairs.csv (frozen v2 main experiment)
    python3 -m cch.off2_repair summarize  # -> c-summary.csv

For each of the 36 instances (validation B's sample), the base packing is
built by the frozen base core S2-TRoff (perimeter/fitness-number/SN, no tower
removal) through the order-agnostic continuation path, and every candidate
window [x0, x0+w) with w = floor(0.2*W) is scored: removal set (half-open
intersection), H_fixed (highest remaining top), qualified := H_fixed < H0,
and hole score (per-column uncovered cells below the skyline).

Main experiment (frozen v2): among qualified windows only, the guided arm
takes the hole-score argmax (ties leftmost); the random arm draws 20 windows
with replacement (seed off2c-v2/<name>), each repaired from an independent
copy of the base layout. The repairer removes the window's items, rebuilds
the skyline per column from the kept placements, and repacks the removed
items (perimeter-descending) via solve_config_from_state. The RAW repair is
geometry-validated before incumbent-keeping (a rejected repair must still be
a legal packing), then the better of {base, repair} is kept. Timing is
serial; the guided arm's select_ms includes hole scoring.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from statistics import mean, median
from time import perf_counter, process_time

from burke_bf import Instance, Item, Placement, load_instance, validate_solution

from .experiment import _lower_bound
from .model import Config, Ordering
from .off2_landscape import CORES, _sample_specs
from .off2_state import solve_config_from_state
from .off2_validation import _bootstrap_ci, _ordering_key
from .shell import _sort_key
from .solver import solve_config

BASE_CORE = CORES["S2-TRoff"]


def _base_packing(instance: Instance) -> tuple[list[Placement], list[int]]:
    """Fresh base run: the degenerate continuation of an empty state."""

    placements, skyline, _ = solve_config_from_state(
        [0] * instance.strip_width, (), list(instance.items), BASE_CORE
    )
    return placements, skyline


def _hole_heights(
    strip_width: int, placements: list[Placement], skyline: list[int]
) -> list[int]:
    """Per column x: uncovered cell count below skyline[x] (buried holes)."""

    intervals: list[list[tuple[int, int]]] = [[] for _ in range(strip_width)]
    for p in placements:
        for x in range(p.x, p.right):
            intervals[x].append((p.y, p.top))
    holes = []
    for x in range(strip_width):
        covered = 0
        cur_s = cur_e = None
        for s, e in sorted(intervals[x]):
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                covered += cur_e - cur_s
                cur_s, cur_e = s, e
        if cur_s is not None:
            covered += cur_e - cur_s
        holes.append(skyline[x] - covered)
    return holes


def _window_rows(
    instance: Instance, placements: list[Placement], skyline: list[int]
) -> tuple[list[dict], int]:
    """Per-candidate-window measurements (design-c v2 C.3)."""

    width = max(int(0.2 * instance.strip_width), 1)
    h0 = max(skyline)
    starts = sorted({p.x for p in placements} | {instance.strip_width - width})
    starts = [x for x in starts if 0 <= x <= instance.strip_width - width]
    holes = _hole_heights(instance.strip_width, placements, skyline)
    rows = []
    for x0 in starts:
        removed = [
            p for p in placements if p.x < x0 + width and p.right > x0
        ]
        removed_ids = {p.item_id for p in removed}
        remaining_top = max(
            (p.top for p in placements if p.item_id not in removed_ids),
            default=0,
        )
        rows.append(
            {
                "x0": x0,
                "removed_n": len(removed),
                "removed_area": sum(p.width * p.height for p in removed),
                "h_fixed": remaining_top,
                "qualified": int(remaining_top < h0),
                "hole_score": sum(holes[x0 : x0 + width]),
            }
        )
    return rows, width


def _precheck(out_dir: str) -> int:
    win_path = f"{out_dir}/c-precheck-windows.csv"
    inst_path = f"{out_dir}/c-precheck-instances.csv"
    inst_rows = []
    with open(win_path, "w", newline="") as win_f:
        win_writer = csv.DictWriter(
            win_f,
            fieldnames=["instance", "x0", "removed_n", "removed_area",
                        "h_fixed", "qualified", "hole_score"],
        )
        win_writer.writeheader()
        for group, path in _sample_specs():
            instance = load_instance(path)
            lb = _lower_bound(instance)
            placements, skyline = _base_packing(instance)
            validate_solution(
                instance,
                _ns(instance, placements, max(skyline)),
            )
            rows, width = _window_rows(instance, placements, skyline)
            for row in rows:
                win_writer.writerow({"instance": instance.name, **row})
            h0 = max(skyline)
            qualified = [r for r in rows if r["qualified"]]
            # Guided pick on the unrestricted candidate set: does it land in
            # the qualified set? (ties: leftmost)
            best_all = max(rows, key=lambda r: (r["hole_score"], -r["x0"]))
            best_qual = (
                max(qualified, key=lambda r: (r["hole_score"], -r["x0"]))
                if qualified
                else None
            )
            inst_rows.append(
                {
                    "instance": instance.name,
                    "group": group,
                    "n": len(instance.items),
                    "lb": lb,
                    "h0": h0,
                    "w": width,
                    "n_cand": len(rows),
                    "n_qualified": len(qualified),
                    "qualified_frac": f"{len(qualified) / len(rows):.4f}",
                    "guided_all_x0": best_all["x0"],
                    "guided_all_qualified": int(bool(best_all["qualified"])),
                    "guided_qual_x0": best_qual["x0"] if best_qual else "",
                    "guided_qual_hole": best_qual["hole_score"] if best_qual else "",
                    "max_hole_score": max(r["hole_score"] for r in rows),
                }
            )
            print(f"  {instance.name:10s} done", file=sys.stderr, flush=True)
    with open(inst_path, "w", newline="") as inst_f:
        writer = csv.DictWriter(inst_f, fieldnames=list(inst_rows[0].keys()))
        writer.writeheader()
        writer.writerows(inst_rows)

    n = len(inst_rows)
    zero = sum(int(r["n_qualified"]) == 0 for r in inst_rows)
    one = sum(int(r["n_qualified"]) == 1 for r in inst_rows)
    many = n - zero - one
    print(f"\n== window precheck (36 instances, w = floor(0.2W)) ==")
    print(f"instances with 0 qualified windows  : {zero}/{n} "
          f"('window family cannot release the height limit')")
    print(f"instances with exactly 1            : {one}/{n} "
          f"('no region-comparison space')")
    print(f"instances with >= 2 (usable)        : {many}/{n}")
    quals = [int(r["n_qualified"]) for r in inst_rows]
    print(f"qualified count: median {median(quals)}, mean {mean(quals):.1f}, "
          f"min {min(quals)}, max {max(quals)}")
    landed = mean(int(r["guided_all_qualified"]) for r in inst_rows)
    print(f"unrestricted hole-guided pick lands in qualified set: "
          f"{landed:.2%} of instances")
    print(f"\nwrote {win_path} and {inst_path}", file=sys.stderr)
    return 0


def _ns(instance: Instance, placements: list[Placement], height: int):
    from types import SimpleNamespace

    return SimpleNamespace(placements=tuple(placements), height=height)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cch.off2_repair",
        description=__doc__.splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("precheck", "run", "summarize", "run-d", "summarize-d",
                 "run-d2", "summarize-d2"):
        child = sub.add_parser(name)
        child.add_argument("--out-dir", default="docs/off2-learn")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.command == "precheck":
        return _precheck(args.out_dir)
    if args.command == "run":
        return _run(args.out_dir)
    if args.command == "run-d":
        return _run_d(args.out_dir)
    if args.command == "summarize-d":
        return _summarize_d(args.out_dir)
    if args.command == "run-d2":
        return _run_d2(args.out_dir)
    if args.command == "summarize-d2":
        return _summarize_d2(args.out_dir)
    return _summarize(args.out_dir)


def _rebuild_skyline(strip_width: int, placements: list[Placement]) -> list[int]:
    """Per-column max top of the kept placements (0 where uncovered)."""

    skyline = [0] * strip_width
    for p in placements:
        for x in range(p.x, p.right):
            if p.top > skyline[x]:
                skyline[x] = p.top
    return skyline


def _repair(
    instance: Instance,
    placements: list[Placement],
    removed: list[Placement],
) -> tuple[list[Placement], list[int]]:
    """Remove ``removed`` and repack those items onto the partial layout.

    Returns the RAW repair ``(placements, skyline)`` (before
    incumbent-keeping); callers validate it before any acceptance decision.
    Removed items are fed in the base core's ordering (perimeter descending
    on normalised dimensions).
    """

    removed_ids = {p.item_id for p in removed}
    kept = [p for p in placements if p.item_id not in removed_ids]
    skyline = _rebuild_skyline(instance.strip_width, kept)
    by_id = {item.item_id: item for item in instance.items}
    items = sorted(
        (by_id[p.item_id] for p in removed),
        key=_ordering_key(BASE_CORE.ordering, rotatable=True),
    )
    new_placements, new_skyline, _ = solve_config_from_state(
        skyline, kept, items, BASE_CORE
    )
    return new_placements, new_skyline


def _run(out_dir: str) -> int:
    """Frozen v2 main experiment: guided vs 20 random qualified windows."""

    repairs_path = f"{out_dir}/c-repairs.csv"
    fields = [
        "instance", "group", "n", "lb", "h0", "w", "n_qualified",
        "arm", "draw_ix", "x0", "removed_n", "removed_area", "h_fixed",
        "hole_score", "guided_degenerate", "h_raw", "h_kept", "gain",
        "select_ms", "repair_ms",
    ]
    from time import perf_counter

    all_rows: list[dict] = []
    started = perf_counter()
    for group, path in _sample_specs():
        instance = load_instance(path)
        lb = _lower_bound(instance)
        placements, skyline = _base_packing(instance)
        h0 = max(skyline)
        validate_solution(instance, _ns(instance, placements, h0))
        rows, width = _window_rows(instance, placements, skyline)
        qualified = [r for r in rows if r["qualified"]]
        if len(qualified) < 2:
            # 0 or 1 qualified window: recorded in the precheck; no
            # region-comparison space, so no paired repair runs.
            print(
                f"  {instance.name:10s} skipped ({len(qualified)} qualified)",
                file=sys.stderr, flush=True,
            )
            continue
        rng = random.Random(f"off2c-v2/{instance.name}")
        arms = []
        best = max(qualified, key=lambda r: (r["hole_score"], -r["x0"]))
        degenerate = int(all(r["hole_score"] == 0 for r in qualified))
        arms.append(("G", 0, best))
        for draw in range(1, 21):
            arms.append(("R", draw, rng.choice(qualified)))
        for arm, draw_ix, win in arms:
            removed = [
                p
                for p in placements
                if p.x < win["x0"] + width and p.right > win["x0"]
            ]
            t0 = perf_counter()
            new_placements, new_skyline = _repair(instance, placements, removed)
            h_raw = max(new_skyline)
            repair_ms = (perf_counter() - t0) * 1000.0
            # Validate the RAW repair before any incumbent decision.
            validate_solution(instance, _ns(instance, new_placements, h_raw))
            h_kept = min(h0, h_raw)
            all_rows.append(
                {
                    "instance": instance.name,
                    "group": group,
                    "n": len(instance.items),
                    "lb": lb,
                    "h0": h0,
                    "w": width,
                    "n_qualified": len(qualified),
                    "arm": arm,
                    "draw_ix": draw_ix,
                    "x0": win["x0"],
                    "removed_n": len(removed),
                    "removed_area": sum(p.width * p.height for p in removed),
                    "h_fixed": win["h_fixed"],
                    "hole_score": win["hole_score"],
                    "guided_degenerate": degenerate if arm == "G" else "",
                    "h_raw": h_raw,
                    "h_kept": h_kept,
                    "gain": f"{(h0 - h_kept) / lb:.6f}",
                    "select_ms": "",
                    "repair_ms": f"{repair_ms:.3f}",
                }
            )
        # Guided-arm selection cost (hole scoring over all qualified
        # windows), measured once, serially.
        t0 = perf_counter()
        holes = _hole_heights(instance.strip_width, placements, skyline)
        _ = max(
            qualified,
            key=lambda r: (sum(holes[r["x0"] : r["x0"] + width]), -r["x0"]),
        )
        select_ms = (perf_counter() - t0) * 1000.0
        for row in all_rows:
            if row["instance"] == instance.name and row["arm"] == "G":
                row["select_ms"] = f"{select_ms:.3f}"
        print(
            f"  {instance.name:10s} done ({perf_counter() - started:.0f}s)",
            file=sys.stderr, flush=True,
        )
    with open(repairs_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {repairs_path}", file=sys.stderr)
    return 0


def _summarize(out_dir: str) -> int:
    rows = list(csv.DictReader(open(f"{out_dir}/c-repairs.csv", newline="")))
    per_instance: dict[str, dict] = {}
    for row in rows:
        entry = per_instance.setdefault(row["instance"], {"G": None, "R": [], "meta": row})
        if row["arm"] == "G":
            entry["G"] = row
        else:
            entry["R"].append(row)

    # Echo the precheck structure counts (reviewer point 1: the run record
    # must state them; the frozen window family was not revised afterwards).
    precheck = list(csv.DictReader(open(f"{out_dir}/c-precheck-instances.csv", newline="")))
    n_zero = sum(int(r["n_qualified"]) == 0 for r in precheck)
    n_one = sum(int(r["n_qualified"]) == 1 for r in precheck)
    n_many = len(precheck) - n_zero - n_one
    assert n_many == len(per_instance), "precheck and run disagree on usable instances"

    summary_rows = []

    def add(section, subject, metric, value, ci_lo="", ci_hi="", note=""):
        summary_rows.append(
            {"section": section, "subject": subject, "metric": metric,
             "value": value, "ci_lo": ci_lo, "ci_hi": ci_hi,
             "n_instances": len(per_instance), "note": note}
        )

    gains_g = []
    gains_r_mean = []
    diffs = []
    quantiles = []
    zero_both = 0
    for name, entry in per_instance.items():
        g_g = float(entry["G"]["gain"])
        r_gains = [float(r["gain"]) for r in entry["R"]]
        gains_g.append(g_g)
        gains_r_mean.append(mean(r_gains))
        diffs.append(g_g - mean(r_gains))
        quantiles.append(mean(rg < g_g for rg in r_gains))
        if g_g == 0 and all(rg == 0 for rg in r_gains):
            zero_both += 1
    point = mean(diffs)
    lo, hi = _bootstrap_ci(diffs)
    add("comparison", "G - mean(R)", "mean_d", f"{point:.6f}", f"{lo:.6f}", f"{hi:.6f}",
        note="LB-normalised gain difference, instance-paired bootstrap")

    if lo > 0:
        verdict = "support-G"
        note = "CI lower bound > 0: guided region selection has value (next stage)"
    elif hi < 0:
        verdict = "support-G-worse"
        note = "CI upper bound < 0: guided selection is worse under current conditions"
    else:
        verdict = "insufficient"
        note = "CI includes zero"
    add("judgment", "G vs R", "verdict", verdict, note=note)

    add("protective", "structure", "qualified_window_counts",
        f"zero={n_zero}, one={n_one}, ge2={n_many} of 36",
        note="from c-precheck-instances.csv (window family unchanged)")
    add("protective", "observable_gain", "mean_gain_G", f"{mean(gains_g):.6f}")
    add("protective", "observable_gain", "mean_gain_R", f"{mean(gains_r_mean):.6f}")
    add("protective", "observable_gain", "instances_both_arms_zero",
        f"{zero_both}/{len(per_instance)}",
        note="if qualified windows exist but both arms are all zero: '当前窗口族与"
             "固定修复器未产生可观测的高度改善，尚不能区分选区信号不足与修复器能力不足'")

    add("aux", "guided_quantile", "mean share of R draws worse than G",
        f"{mean(quantiles):.4f}")
    degenerate = mean(
        int(e["G"]["guided_degenerate"]) for e in per_instance.values()
    )
    add("aux", "guided_degenerate", "proportion", f"{degenerate:.4f}")
    for field in ("removed_n", "removed_area"):
        g_vals = [int(e["G"][field]) for e in per_instance.values()]
        r_vals = [mean(int(r[field]) for r in e["R"]) for e in per_instance.values()]
        add("aux", "removal_size", f"{field} G vs R(mean)",
            f"{mean(g_vals):.2f} vs {mean(r_vals):.2f}")
    g_sel = [float(e["G"]["select_ms"]) for e in per_instance.values()]
    g_rep = [float(e["G"]["repair_ms"]) for e in per_instance.values()]
    r_rep = [float(r["repair_ms"]) for e in per_instance.values() for r in e["R"]]
    add("aux", "timing_serial", "G select_ms (incl. hole scoring)", f"{mean(g_sel):.3f}")
    add("aux", "timing_serial", "G repair_ms", f"{mean(g_rep):.3f}")
    add("aux", "timing_serial", "R repair_ms", f"{mean(r_rep):.3f}")

    with open(f"{out_dir}/c-summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"usable instances: {len(per_instance)}/36 "
          f"(precheck: zero-qualified {n_zero}, one {n_one}, >=2 {n_many})")
    print(f"mean gain  G={mean(gains_g):.4f}  R(mean)={mean(gains_r_mean):.4f}  "
          f"(LB-normalised)")
    print(f"paired d = G - mean(R): {point:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"verdict: {verdict} — {note}")
    print(f"both arms all-zero on {zero_both}/{len(per_instance)} usable instances")
    print(f"guided gain beats share of random draws: {mean(quantiles):.3f}")
    print(f"\nwrote {out_dir}/c-summary.csv", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- validation D

T_CHAIN = 20
RLS_BUDGETS = (20, 100)
CHAIN_SEEDS = (1, 2, 3, 4, 5)


def _holdout_specs() -> list[tuple[str, str]]:
    """Held-out families (design-d D.1): prospective C/D groups, n=300."""

    return [
        (f"PV-{g}", f"data/prospective/pv-{g}-300-{r:02d}.ins2D")
        for g in "cd"
        for r in range(1, 6)
    ]


def _chain_walk(
    instance: Instance,
    placements: list[Placement],
    skyline: list[int],
    arm: str,
    lb: int,
    seed: str | None,
    tabu_mode: str = "permanent",
) -> tuple[list[dict], list[Placement], list[int], int, int]:
    """One repair walk: (step rows, final placements, final skyline, improving,
    degenerate count).

    ``tabu_mode="permanent"`` (v1): a tried x0 is banned for the whole walk.
    ``"layout"`` (v2, reviewed rule): no repeat within the *current* layout —
    a non-improving repair marks the window tried; an accepted strict
    improvement clears the tried set and windows are regenerated.
    """

    from time import perf_counter

    rng = random.Random(seed) if seed is not None else None
    placements = list(placements)
    skyline = list(skyline)
    h_cur = max(skyline)
    tried: set[int] = set()
    step_rows = []
    improving = 0
    degenerate = 0
    for step in range(T_CHAIN):
        if h_cur <= lb:
            break
        t0 = perf_counter()
        rows, width = _window_rows(instance, placements, skyline)
        cand = [r for r in rows if r["qualified"] and r["x0"] not in tried]
        if arm == "G" and cand:
            win = max(cand, key=lambda r: (r["hole_score"], -r["x0"]))
            degen = int(all(r["hole_score"] == 0 for r in cand))
            degenerate += degen
        elif cand:
            win = rng.choice(cand)
            degen = ""
        select_ms = (perf_counter() - t0) * 1000.0
        if not cand:
            break
        tried.add(win["x0"])
        removed = [
            p for p in placements if p.x < win["x0"] + width and p.right > win["x0"]
        ]
        t1 = perf_counter()
        new_placements, new_skyline = _repair(instance, placements, removed)
        repair_ms = (perf_counter() - t1) * 1000.0
        h_raw = max(new_skyline)
        validate_solution(instance, _ns(instance, new_placements, h_raw))
        if h_raw < h_cur:
            placements, skyline, h_cur = new_placements, new_skyline, h_raw
            improving += 1
            if tabu_mode == "layout":
                tried.clear()
        step_rows.append(
            {
                "step": step,
                "x0": win["x0"],
                "removed_n": len(removed),
                "hole_score": win["hole_score"],
                "qualified_n": len(cand),
                "degenerate_step": degen,
                "h_raw": h_raw,
                "h_kept": h_cur,
                "gain_so_far": "",  # filled by caller (needs h0)
                "select_ms": f"{select_ms:.3f}",
                "repair_ms": f"{repair_ms:.3f}",
            }
        )
    return step_rows, placements, skyline, improving, degenerate


def _rls_budget(
    instance: Instance, config: Config, budget_max: int, seed: int, lb: int
) -> tuple[dict[int, tuple[int, float]], int]:
    """``cch.shell.random_ls`` with an evaluation budget instead of wall clock.

    Snapshots ``(incumbent, elapsed_ms)`` at each budget in ``RLS_BUDGETS``
    within one run. The frozen engine is safe for this core (fitness-number
    has no order-sensitive early break).
    """

    from time import perf_counter

    items = list(instance.items)
    rng = random.Random(seed)
    n = len(items)
    evals = 0
    started = perf_counter()

    def pack(sequence: list[Item]) -> int:
        nonlocal evals
        _placements, skyline, _moves = solve_config(instance, config, sequence=sequence)
        evals += 1
        return max(skyline)

    initial: list[tuple[int, str, list[Item]]] = []
    for ordering in Ordering:
        sequence = sorted(items, key=lambda item: _sort_key(ordering, item))
        height = pack(sequence)
        initial.append((height, ordering.value, sequence))
    initial.sort(key=lambda entry: (entry[0], entry[1]))
    best = initial[0][0]

    snapshot: dict[int, tuple[int, float]] = {}

    def snap() -> None:
        for budget in RLS_BUDGETS:
            if budget not in snapshot and evals >= budget:
                snapshot[budget] = (best, (perf_counter() - started) * 1000.0)

    snap()
    while evals < budget_max and best > lb:
        for _h, _name, sequence in initial:
            if evals >= budget_max or best <= lb:
                break
            current = sequence
            height = pack(current)
            best = min(best, height)
            for _i in range(n):
                if evals >= budget_max or best <= lb:
                    break
                a, b = rng.sample(range(n), 2)
                candidate = list(current)
                candidate[a], candidate[b] = candidate[b], candidate[a]
                new_height = pack(candidate)
                if new_height <= height:
                    height, current = new_height, candidate
                    best = min(best, new_height)
            snap()
        snap()
    for budget in RLS_BUDGETS:
        snapshot.setdefault(budget, (best, (perf_counter() - started) * 1000.0))
    return snapshot, evals


def _run_d(out_dir: str) -> int:
    """Validation D: guided/random repair chains + budgeted RLS reference."""

    from time import perf_counter

    from .rls_multiseed import CORES as RLS_CORES

    rls_core = RLS_CORES["stage2-best"]
    steps_path = f"{out_dir}/d-chain-steps.csv"
    runs_path = f"{out_dir}/d-runs.csv"
    step_fields = ["instance", "set", "arm", "seed", "step", "x0", "removed_n",
                   "hole_score", "qualified_n", "degenerate_step", "h_raw",
                   "h_kept", "gain_so_far"]
    run_fields = ["instance", "set", "arm", "seed", "budget", "h0", "lb",
                  "h_final", "gain", "steps_done", "evals", "improving_steps",
                  "degenerate_steps", "walk_ms"]
    started = perf_counter()
    with open(steps_path, "w", newline="") as steps_f, open(
        runs_path, "w", newline=""
    ) as runs_f:
        steps_writer = csv.DictWriter(
            steps_f, fieldnames=step_fields, extrasaction="ignore"
        )
        runs_writer = csv.DictWriter(runs_f, fieldnames=run_fields)
        steps_writer.writeheader()
        runs_writer.writeheader()
        for set_name, specs in (("dev", _sample_specs()), ("holdout", _holdout_specs())):
            for group, path in specs:
                instance = load_instance(path)
                lb = _lower_bound(instance)
                placements, skyline = _base_packing(instance)
                h0 = max(skyline)
                validate_solution(instance, _ns(instance, placements, h0))
                for arm, seeds in (("G", [None]), ("R", CHAIN_SEEDS)):
                    for seed in seeds:
                        t0 = perf_counter()
                        rows, fin_p, fin_sky, improving, degen = _chain_walk(
                            instance, placements, skyline, arm, lb,
                            f"off2d-v1/{instance.name}/{seed}" if seed else None,
                        )
                        walk_ms = (perf_counter() - t0) * 1000.0
                        for row in rows:
                            row["gain_so_far"] = f"{(h0 - row['h_kept']) / lb:.6f}"
                            steps_writer.writerow(
                                {"instance": instance.name, "set": set_name,
                                 "arm": arm, "seed": seed or "", **row}
                            )
                        h_final = max(fin_sky)
                        runs_writer.writerow(
                            {"instance": instance.name, "set": set_name,
                             "arm": arm, "seed": seed or "", "budget": T_CHAIN,
                             "h0": h0, "lb": lb, "h_final": h_final,
                             "gain": f"{(h0 - h_final) / lb:.6f}",
                             "steps_done": len(rows), "evals": len(rows),
                             "improving_steps": improving,
                             "degenerate_steps": degen,
                             "walk_ms": f"{walk_ms:.1f}"}
                        )
                for seed in CHAIN_SEEDS:
                    snapshot, evals = _rls_budget(
                        instance, rls_core, max(RLS_BUDGETS), seed, lb
                    )
                    for budget in RLS_BUDGETS:
                        h_final, budget_ms = snapshot[budget]
                        runs_writer.writerow(
                            {"instance": instance.name, "set": set_name,
                             "arm": "RLS", "seed": seed, "budget": budget,
                             "h0": h0, "lb": lb, "h_final": h_final,
                             "gain": f"{(h0 - h_final) / lb:.6f}",
                             "steps_done": "", "evals": min(evals, budget),
                             "improving_steps": "", "degenerate_steps": "",
                             "walk_ms": f"{budget_ms:.1f}"}
                        )
                steps_f.flush()
                runs_f.flush()
                print(
                    f"  {set_name:7s} {instance.name:10s} done "
                    f"({perf_counter() - started:.0f}s)",
                    file=sys.stderr, flush=True,
                )
    print(f"wrote {steps_path} and {runs_path}", file=sys.stderr)
    return 0

def _summarize_d(out_dir: str) -> int:
    runs = list(csv.DictReader(open(f"{out_dir}/d-runs.csv", newline="")))
    per: dict[tuple[str, str], dict] = {}
    for row in runs:
        key = (row["set"], row["instance"])
        entry = per.setdefault(
            key,
            {"G": None, "R": [], "RLS20": [], "RLS100": [],
             "h0": int(row["h0"]), "lb": int(row["lb"])},
        )
        gain = float(row["gain"])
        if row["arm"] == "G":
            entry["G"] = row
        elif row["arm"] == "R":
            entry["R"].append(gain)
        elif row["budget"] == "20":
            entry["RLS20"].append(gain)
        else:
            entry["RLS100"].append(gain)

    summary_rows = []

    def add(set_name, section, subject, metric, value, ci_lo="", ci_hi="", note=""):
        summary_rows.append(
            {"set": set_name, "section": section, "subject": subject,
             "metric": metric, "value": value, "ci_lo": ci_lo, "ci_hi": ci_hi,
             "note": note}
        )

    for set_name in ("dev", "holdout"):
        cells = [(k, e) for k, e in sorted(per.items()) if k[0] == set_name]
        g = [float(e["G"]["gain"]) for _k, e in cells]
        r = [mean(e["R"]) for _k, e in cells]
        rls20 = [mean(e["RLS20"]) for _k, e in cells]
        rls100 = [mean(e["RLS100"]) for _k, e in cells]
        n = len(cells)

        d_gr = [a - b for a, b in zip(g, r)]
        point = mean(d_gr)
        lo, hi = _bootstrap_ci(d_gr)
        add(set_name, "compound", "G - mean(R)", "mean_d", f"{point:.6f}",
            f"{lo:.6f}", f"{hi:.6f}", note="guidance compounding test (i)")

        d_rls = [a - b for a, b in zip(g, rls20)]
        point2 = mean(d_rls)
        lo2, hi2 = _bootstrap_ci(d_rls)
        add(set_name, "bar", "G - RLS@20", "mean_d", f"{point2:.6f}",
            f"{lo2:.6f}", f"{hi2:.6f}",
            note="competitive-bar test (ii): support needs point >= 0")

        add(set_name, "means", "gain", "G / R / RLS20 / RLS100",
            f"{mean(g):.5f} / {mean(r):.5f} / {mean(rls20):.5f} / "
            f"{mean(rls100):.5f}")

        base_at_lb = sum(1 for _k, e in cells if e["h0"] == e["lb"])
        zero_g = sum(1 for _k, e in cells if float(e["G"]["gain"]) == 0)
        zero_both = sum(
            1 for _k, e in cells
            if float(e["G"]["gain"]) == 0 and mean(e["R"]) == 0
        )
        add(set_name, "protective", "base_at_LB", "count", f"{base_at_lb}/{n}")
        add(set_name, "protective", "zero_gain", "G / both",
            f"{zero_g}/{n} / {zero_both}/{n}",
            note="if both arms zero: cannot separate region signal from "
                 "repairer capability")

        steps_done = [float(e["G"]["steps_done"]) for _k, e in cells]
        degen = [float(e["G"]["degenerate_steps"]) for _k, e in cells]
        walk_ms = [float(e["G"]["walk_ms"]) for _k, e in cells]
        add(set_name, "aux", "steps_done", "G mean", f"{mean(steps_done):.2f}")
        add(set_name, "aux", "degenerate_steps", "G mean", f"{mean(degen):.2f}")
        add(set_name, "aux", "walk_ms", "G mean (serial)", f"{mean(walk_ms):.1f}")

        verdict = "support" if (lo > 0 and point2 >= 0) else "stop"
        note = ("(i) compounding CI lower > 0 and (ii) G not worse than "
                "RLS@20 on point estimate" if verdict == "support"
                else "hard gate (design-d D.0) not met; ③ stops / becomes a "
                     "mechanism note")
        add(set_name, "judgment", "verdict", "hard-gate", verdict, note=note)

    steps = list(csv.DictReader(open(f"{out_dir}/d-chain-steps.csv", newline="")))
    walks: dict[tuple[str, str, str, str], dict[int, float]] = {}
    for r in steps:
        wkey = (r["set"], r["instance"], r["arm"], r["seed"])
        walks.setdefault(wkey, {})[int(r["step"])] = float(r["gain_so_far"])
    for set_name in ("dev", "holdout"):
        n_inst = sum(1 for k in per if k[0] == set_name)
        for arm in ("G", "R"):
            # Denominator includes zero-step walks (absent from the steps CSV);
            # LOCF carries a stopped walk's last gain forward.
            denom = n_inst if arm == "G" else n_inst * len(CHAIN_SEEDS)
            vals = []
            for target in (1, 5, 10, 20):
                total = 0.0
                for (s2, _inst, a2, _seed), gains in walks.items():
                    if s2 != set_name or a2 != arm:
                        continue
                    upto = [g for st, g in gains.items() if st <= target - 1]
                    total += upto[-1] if upto else 0.0
                vals.append(
                    f"step{target}={total / denom:.5f}" if denom else f"step{target}=NA"
                )
            add(set_name, "aux", "trajectory_locf", arm, "; ".join(vals))

    with open(f"{out_dir}/d-summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for set_name in ("dev", "holdout"):
        print(f"\n===== {set_name} =====")
        for row in summary_rows:
            if row["set"] != set_name:
                continue
            if row["section"] in ("compound", "bar"):
                print(f"  {row['subject']:14s} {row['value']:>9s} "
                      f"[{row['ci_lo']:>9s}, {row['ci_hi']:>9s}]")
            elif row["section"] == "means":
                print(f"  gains G/R/RLS20/RLS100: {row['value']}")
            elif row["section"] == "protective":
                print(f"  {row['subject']:12s} {row['metric']:12s} {row['value']}")
            elif row["section"] == "judgment":
                print(f"  VERDICT: {row['value']} — {row['note']}")
            elif row["section"] == "aux" and row["subject"] == "trajectory_locf":
                print(f"  trajectory(LOCF) {row['metric']}: {row['value']}")
    print(f"\nwrote {out_dir}/d-summary.csv", file=sys.stderr)
    return 0


# ------------------------------------------------------------- validation D v2


def _rls_wall(
    instance: Instance,
    config: Config,
    time_limit_s: float,
    seed: int,
    lb: int,
    clock: str = "wall",
) -> tuple[int, int, int, float]:
    """random_ls replica with a serial budget (design-d v2 D.3).

    ``clock="wall"`` measures wall time (perf_counter); ``"cpu"`` measures
    process CPU time (process_time), which is robust to load from other
    processes. Returns (best, initial_best, evals, elapsed_ms).
    """

    timer = perf_counter if clock == "wall" else process_time
    items = list(instance.items)
    rng = random.Random(seed)
    n = len(items)
    started = timer()
    evals = 0

    def pack(seq: list[Item]) -> int:
        nonlocal evals
        _p, skyline, _m = solve_config(instance, config, sequence=seq)
        evals += 1
        return max(skyline)

    initial: list[tuple[int, str, list[Item]]] = []
    for ordering in Ordering:
        sequence = sorted(items, key=lambda item: _sort_key(ordering, item))
        initial.append((pack(sequence), ordering.value, sequence))
    initial.sort(key=lambda entry: (entry[0], entry[1]))
    initial_best = best = initial[0][0]
    while timer() - started < time_limit_s and best > lb:
        for _h, _name, sequence in initial:
            if timer() - started > time_limit_s or best <= lb:
                break
            current = sequence
            height = pack(current)
            best = min(best, height)
            for _i in range(n):
                if timer() - started > time_limit_s or best <= lb:
                    break
                a, b = rng.sample(range(n), 2)
                candidate = list(current)
                candidate[a], candidate[b] = candidate[b], candidate[a]
                new_height = pack(candidate)
                if new_height <= height:
                    height, current = new_height, candidate
                    best = min(best, new_height)
    return best, initial_best, evals, (timer() - started) * 1000.0


def _run_d2(out_dir: str) -> int:
    """Validation D v2: layout-tabu chains + call-count and wall-clock RLS arms."""

    from time import perf_counter

    from .rls_multiseed import CORES as RLS_CORES

    rls_core = RLS_CORES["stage2-best"]
    steps_path = f"{out_dir}/d2-chain-steps.csv"
    runs_path = f"{out_dir}/d2-runs.csv"
    step_fields = ["instance", "set", "arm", "seed", "step", "x0", "removed_n",
                   "hole_score", "qualified_n", "degenerate_step", "h_raw",
                   "h_kept", "gain_so_far", "select_ms", "repair_ms"]
    run_fields = ["instance", "set", "arm", "seed", "budget", "h0", "lb",
                  "initial_h", "h_final", "gain", "steps_done", "evals",
                  "improving_steps", "degenerate_steps", "walk_ms"]
    started = perf_counter()
    with open(steps_path, "w", newline="") as steps_f, open(
        runs_path, "w", newline=""
    ) as runs_f:
        steps_writer = csv.DictWriter(steps_f, fieldnames=step_fields)
        runs_writer = csv.DictWriter(runs_f, fieldnames=run_fields)
        steps_writer.writeheader()
        runs_writer.writeheader()
        for set_name, specs in (("dev", _sample_specs()), ("holdout", _holdout_specs())):
            for group, path in specs:
                instance = load_instance(path)
                lb = _lower_bound(instance)
                placements, skyline = _base_packing(instance)
                h0 = max(skyline)
                validate_solution(instance, _ns(instance, placements, h0))
                # Initial quality of the RLS arm (best of 6 orderings), for
                # reporting; chains start at h0.
                rls_initial = min(
                    max(solve_config(
                        instance, rls_core,
                        sequence=sorted(
                            list(instance.items),
                            key=lambda item: _sort_key(o, item),
                        ),
                    )[1])
                    for o in Ordering
                )
                g_walk_ms = 0.0
                g_walk_cpu_ms = 0.0
                for arm, seeds in (("G", [None]), ("R", CHAIN_SEEDS)):
                    for seed in seeds:
                        t0 = perf_counter()
                        t0c = process_time()
                        rows, _fp, fin_sky, improving, degen = _chain_walk(
                            instance, placements, skyline, arm, lb,
                            f"off2d-v2/{instance.name}/{seed}" if seed else None,
                            tabu_mode="layout",
                        )
                        walk_ms = (perf_counter() - t0) * 1000.0
                        walk_cpu_ms = (process_time() - t0c) * 1000.0
                        if arm == "G":
                            g_walk_ms = walk_ms
                            g_walk_cpu_ms = walk_cpu_ms
                        for row in rows:
                            row["gain_so_far"] = f"{(h0 - row['h_kept']) / lb:.6f}"
                            steps_writer.writerow(
                                {"instance": instance.name, "set": set_name,
                                 "arm": arm, "seed": seed or "", **row}
                            )
                        h_final = max(fin_sky)
                        runs_writer.writerow(
                            {"instance": instance.name, "set": set_name,
                             "arm": arm, "seed": seed or "", "budget": T_CHAIN,
                             "h0": h0, "lb": lb, "initial_h": h0,
                             "h_final": h_final,
                             "gain": f"{(h0 - h_final) / lb:.6f}",
                             "steps_done": len(rows), "evals": len(rows),
                             "improving_steps": improving,
                             "degenerate_steps": degen,
                             "walk_ms": f"{walk_ms:.1f}"}
                        )
                for seed in CHAIN_SEEDS:
                    snapshot, evals = _rls_budget(
                        instance, rls_core, max(RLS_BUDGETS), seed, lb
                    )
                    for budget in RLS_BUDGETS:
                        h_final, budget_ms = snapshot[budget]
                        runs_writer.writerow(
                            {"instance": instance.name, "set": set_name,
                             "arm": "RLS", "seed": seed, "budget": budget,
                             "h0": h0, "lb": lb, "initial_h": rls_initial,
                             "h_final": h_final,
                             "gain": f"{(h0 - h_final) / lb:.6f}",
                             "steps_done": "", "evals": min(evals, budget),
                             "improving_steps": "", "degenerate_steps": "",
                             "walk_ms": f"{budget_ms:.1f}"}
                        )
                # Wall-clock arm: serial budget = this instance's G-walk time.
                for seed in CHAIN_SEEDS:
                    best, init_b, evals, elapsed_ms = _rls_wall(
                        instance, rls_core, g_walk_ms / 1000.0, seed, lb
                    )
                    runs_writer.writerow(
                        {"instance": instance.name, "set": set_name,
                         "arm": "RLS-wall", "seed": seed,
                         "budget": f"{g_walk_ms:.1f}ms",
                         "h0": h0, "lb": lb, "initial_h": init_b,
                         "h_final": best,
                         "gain": f"{(h0 - best) / lb:.6f}",
                         "steps_done": "", "evals": evals,
                         "improving_steps": "", "degenerate_steps": "",
                         "walk_ms": f"{elapsed_ms:.1f}"}
                    )
                # CPU-time twin arm (v2.1): budget = G walk's process CPU time,
                # robust to load from other processes on this machine.
                for seed in CHAIN_SEEDS:
                    best, init_b, evals, elapsed_ms = _rls_wall(
                        instance, rls_core, g_walk_cpu_ms / 1000.0, seed, lb,
                        clock="cpu",
                    )
                    runs_writer.writerow(
                        {"instance": instance.name, "set": set_name,
                         "arm": "RLS-cpu", "seed": seed,
                         "budget": f"{g_walk_cpu_ms:.1f}ms-cpu",
                         "h0": h0, "lb": lb, "initial_h": init_b,
                         "h_final": best,
                         "gain": f"{(h0 - best) / lb:.6f}",
                         "steps_done": "", "evals": evals,
                         "improving_steps": "", "degenerate_steps": "",
                         "walk_ms": f"{elapsed_ms:.1f}"}
                    )
                steps_f.flush()
                runs_f.flush()
                print(
                    f"  {set_name:7s} {instance.name:10s} done "
                    f"({perf_counter() - started:.0f}s)",
                    file=sys.stderr, flush=True,
                )
    print(f"wrote {steps_path} and {runs_path}", file=sys.stderr)
    return 0


def _summarize_d2(out_dir: str) -> int:
    """v2 summary: persistence + amplification + competitive refs + decision."""

    runs = list(csv.DictReader(open(f"{out_dir}/d2-runs.csv", newline="")))
    steps = list(csv.DictReader(open(f"{out_dir}/d2-chain-steps.csv", newline="")))
    per: dict[tuple[str, str], dict] = {}
    for row in runs:
        key = (row["set"], row["instance"])
        entry = per.setdefault(
            key,
            {"G": None, "R": [], "RLS20": [], "RLS100": [], "RLS-wall": [],
             "RLS-cpu": [],
             "h0": int(row["h0"]), "lb": int(row["lb"]),
             "initial_h": int(row["initial_h"])},
        )
        gain = float(row["gain"])
        if row["arm"] == "G":
            entry["G"] = row
        elif row["arm"] == "R":
            entry["R"].append(gain)
        elif row["arm"] == "RLS-wall":
            entry["RLS-wall"].append(gain)
        elif row["arm"] == "RLS-cpu":
            entry["RLS-cpu"].append(gain)
        elif row["budget"] == "20":
            entry["RLS20"].append(gain)
        else:
            entry["RLS100"].append(gain)

    # Per-walk LOCF trajectories for the amplification test.
    walks: dict[tuple[str, str, str, str], dict[int, float]] = {}
    for r in steps:
        walks.setdefault((r["set"], r["instance"], r["arm"], r["seed"]), {})[
            int(r["step"])
        ] = float(r["gain_so_far"])

    def locf(set_name, instance, arm, seed, step):
        gains = walks.get((set_name, instance, arm, seed), {})
        upto = [g for st, g in sorted(gains.items()) if st <= step]
        return upto[-1] if upto else 0.0

    summary_rows = []

    def add(set_name, section, subject, metric, value, lo="", hi="", note=""):
        summary_rows.append(
            {"set": set_name, "section": section, "subject": subject,
             "metric": metric, "value": value, "ci_lo": lo, "ci_hi": hi,
             "note": note}
        )

    for set_name in ("dev", "holdout"):
        cells = [(k, e) for k, e in sorted(per.items()) if k[0] == set_name]
        n = len(cells)
        g = [float(e["G"]["gain"]) for _k, e in cells]
        r = [mean(e["R"]) for _k, e in cells]
        rls20 = [mean(e["RLS20"]) for _k, e in cells]
        rlsw = [mean(e["RLS-wall"]) for _k, e in cells]
        rlsc = [mean(e["RLS-cpu"]) for _k, e in cells]

        d_gr = [a - b for a, b in zip(g, r)]
        p_gr = mean(d_gr)
        lo, hi = _bootstrap_ci(d_gr)
        add(set_name, "persistence", "G - mean(R)", "mean_d", f"{p_gr:.6f}",
            f"{lo:.6f}", f"{hi:.6f}",
            note="'multi-step advantage persists' if CI lower > 0")

        # Amplification: step-1 advantage vs final advantage (paired).
        a1, aT = [], []
        for (_s, name), e in cells:
            g1 = locf(_s, name, "G", "", 0)
            r1 = mean(locf(_s, name, "R", str(seed), 0) for seed in CHAIN_SEEDS)
            gt = float(e["G"]["gain"])
            rt = mean(e["R"])
            a1.append(g1 - r1)
            aT.append(gt - rt)
        d_amp = [b - a for a, b in zip(a1, aT)]
        p_amp = mean(d_amp)
        lo_a, hi_a = _bootstrap_ci(d_amp)
        add(set_name, "amplification", "advantage@final - @step1", "mean_d",
            f"{p_amp:.6f}", f"{lo_a:.6f}", f"{hi_a:.6f}",
            note="only a positive CI justifies 'amplified'; else persist/decay")

        for label, ref in (("RLS@20(calls)", rls20), ("RLS-wall", rlsw),
                           ("RLS-cpu", rlsc)):
            d_c = [a - b for a, b in zip(g, ref)]
            p_c = mean(d_c)
            lo_c, hi_c = _bootstrap_ci(d_c)
            add(set_name, "competitive", f"G - {label}", "mean_d",
                f"{p_c:.6f}", f"{lo_c:.6f}", f"{hi_c:.6f}",
                note="point >= 0 reads 'not worse on average', not "
                     "statistical non-inferiority")

        base_gap = mean((e["h0"] - e["lb"]) / e["lb"] for _k, e in cells)
        init_gap = mean((e["initial_h"] - e["lb"]) / e["lb"] for _k, e in cells)
        add(set_name, "initial_quality", "gap", "chain-base / RLS-initial",
            f"{base_gap:.5f} / {init_gap:.5f}")
        add(set_name, "means", "gain", "G / R / RLS20 / RLS100 / RLS-wall / RLS-cpu",
            f"{mean(g):.5f} / {mean(r):.5f} / {mean(rls20):.5f} / "
            f"{mean([mean(e['RLS100']) for _k, e in cells]):.5f} / {mean(rlsw):.5f} / "
            f"{mean(rlsc):.5f}")

        zero_g = sum(1 for _k, e in cells if float(e["G"]["gain"]) == 0)
        zero_both = sum(
            1 for _k, e in cells
            if float(e["G"]["gain"]) == 0 and mean(e["R"]) == 0
        )
        add(set_name, "protective", "zero_gain", "G / both",
            f"{zero_g}/{n} / {zero_both}/{n}",
            note="cannot separate region signal from repairer capability")

        g_sel = g_rep = 0.0
        g_steps = 0
        for r_ in steps:
            if r_["set"] == set_name and r_["arm"] == "G":
                g_sel += float(r_["select_ms"])
                g_rep += float(r_["repair_ms"])
                g_steps += 1
        walk_ms = [float(e["G"]["walk_ms"]) for _k, e in cells]
        add(set_name, "costs", "chain", "G step select/repair ms (mean)",
            f"{g_sel / max(g_steps, 1):.3f} / {g_rep / max(g_steps, 1):.3f}")
        add(set_name, "costs", "chain", "G walk ms (mean)",
            f"{mean(walk_ms):.1f}")
        wall_evals = [
            mean(float(rr["evals"]) for rr in runs
                 if rr["set"] == set_name and rr["instance"] == name
                 and rr["arm"] == "RLS-wall")
            for _k, e in cells for name in [_k[1]]
        ]
        add(set_name, "costs", "RLS-wall", "evals within G-walk time (mean)",
            f"{mean(wall_evals):.1f}")
        cpu_evals = [
            mean(float(rr["evals"]) for rr in runs
                 if rr["set"] == set_name and rr["instance"] == name
                 and rr["arm"] == "RLS-cpu")
            for _k, e in cells for name in [_k[1]]
        ]
        add(set_name, "costs", "RLS-cpu", "evals within G-walk CPU time (mean)",
            f"{mean(cpu_evals):.1f}")

        persist = lo > 0
        competitive_wall = mean([a - b for a, b in zip(g, rlsw)]) >= 0
        competitive_cpu = mean([a - b for a, b in zip(g, rlsc)]) >= 0
        # The CPU-budgeted arm is the load-robust one; the wall arm is kept
        # as reference only (v2.1, machine-load contamination concern).
        competitive = competitive_cpu
        if persist and competitive:
            case = "case1-formal-development-candidate"
        elif persist:
            case = "case2-keep-region-value-consider-repairer-check"
        else:
            case = "case3-pause-current-chain-scheme-not-a-refutation"
        add(set_name, "decision", "four-case table", "case", case,
            note=f"persist(CI lo>0)={persist}; competitive point>=0: "
                 f"cpu={competitive_cpu} (decisive), wall={competitive_wall} "
                 f"(reference, load-sensitive)")

    with open(f"{out_dir}/d2-summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for set_name in ("dev", "holdout"):
        print(f"\n===== {set_name} =====")
        for row in summary_rows:
            if row["set"] != set_name:
                continue
            if row["section"] in ("persistence", "amplification", "competitive"):
                print(f"  [{row['section']:12s}] {row['subject']:26s} "
                      f"{row['value']:>9s} [{row['ci_lo']:>9s}, {row['ci_hi']:>9s}]")
            elif row["section"] in ("initial_quality", "means"):
                print(f"  {row['subject']:24s} {row['value']}")
            elif row["section"] == "protective":
                print(f"  zero-gain {row['metric']:10s} {row['value']}")
            elif row["section"] == "costs":
                print(f"  cost {row['subject']:9s} {row['metric']}: {row['value']}")
            elif row["section"] == "decision":
                print(f"  DECISION: {row['value']}  ({row['note']})")
    print(f"\nwrote {out_dir}/d2-summary.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
