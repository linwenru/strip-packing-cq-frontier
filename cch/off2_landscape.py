"""OFF-2 validation B (core x neighbourhood interaction diagnostic), static arm.

Protocol: the companion design documents (dated, in the data package). Subcommands:

    python3 -m cch.off2_landscape run        # serial (timing arm) -> b-perturbations.csv + b-crosscore.csv
    python3 -m cch.off2_landscape summarize  # -> b-summary.csv

Operationalisations fixed here (2026-09-17, within the frozen text; no
protocol change):

- Chain reading: the 200 predetermined perturbations form an open-loop walk
  over the sequence (each step's "before" state is the previous step's
  "after"); all cores face the identical sequence at every step. Chosen
  because the local-neutrality metric ("one perturbation, layout unchanged")
  and the v1 landscape framing are defined per consecutive pair.
- One schedule per instance (seed ``off2b-sched-v1/<name>``): 80 swaps /
  80 inserts / 40 reversals, generated in that allocation and then shuffled;
  positions uniform over the (constant) sequence length; reversal length
  uniform 2-5. Invalid ops (swap/insert of a position with itself,
  degenerate reversals) are kept, evaluated, and recorded with delta_h=0.
- Main arm: every core receives the same width-descending normalised base
  sequence. Auxiliary arm: each core receives its own default ordering.
- Layout divergence: Jaccard distance over (item_id, x, y, w, h) tuples;
  the physical (label-free) comparison uses the (x, y, w, h) coordinate
  multiset; displacement = per-item Manhattan distance, strip-width
  normalised, averaged over all items; "moved" also counts orientation
  changes.
- Validity: completeness asserted every evaluation; full
  ``validate_solution`` on step 0, every 25th step, and the final step.
- Initial-gap-close sub-analysis: instances whose four cores' initial
  heights on the arm's base sequence span <= 1% of LB.
- Intervals: instance-level paired bootstrap, 10,000 reps, seed 20260917
  (same method and seed as validation A, per v2.1).
- Cores reuse ``cch/rls_multiseed.py`` CORES (B.6): BF = classic-bf,
  S1 = stage1-best, S2 = stage2-best, S2-TRoff = S2 without tower removal.
- Timing: the whole run is serial (single process); eval_ms is per
  packing evaluation.
- Packing goes through ``off2_state``'s loop (``_select_any_order``): the
  frozen engine's widest-fit early break is unsound on the permuted
  sequences of a perturbation walk (found by code inspection 2026-09-17;
  the pre-fix CSVs live in git history under commit baf7aaa). Cores
  S1/S2/S2-TRoff are byte-identical between the two pipelines (their
  selection rules have no order-sensitive break); only BF rows changed.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from statistics import mean
from time import perf_counter

from burke_bf import Instance, Item, load_instance

from .experiment import _instance_paths, _lower_bound
from .model import Config, Ordering
from .off2_state import solve_config_from_state
from .off2_validation import _bootstrap_ci, _ordering_key, _validate
from .rls_multiseed import CORES as RLS_CORES

CORES = {
    "BF": RLS_CORES["classic-bf"],
    "S1": RLS_CORES["stage1-best"],
    "S2": RLS_CORES["stage2-best"],
    "S2-TRoff": replace(RLS_CORES["stage2-best"], tower_removal=False),
}
CROSS_PAIRS = (("S2", "BF"), ("S1", "BF"), ("S2", "S1"), ("S2", "S2-TRoff"))
K_TOTAL = 200
N_SWAP, N_INSERT, N_REVERSAL = 80, 80, 40
ARMS = ("main", "own-ordering")


def _sample_specs() -> list[tuple[str, str]]:
    """Stratified sample (B.1): 10 per family with n <= 500, plus PV A/B n=300."""

    rng = random.Random("off2b-sample-v1")
    specs: list[tuple[str, str]] = []
    for data_set, group in (("data/c", "C"), ("data/bkw", "BKW"), ("data/nt", "NT")):
        eligible = [
            path
            for ds, path in _instance_paths()
            if ds == data_set and len(load_instance(path).items) <= 500
        ]
        chosen = sorted(rng.sample(eligible, k=min(10, len(eligible))))
        specs.extend((group, path) for path in chosen)
    specs += [
        ("PV-A", f"data/prospective/pv-a-300-{r:02d}.ins2D") for r in (1, 2, 3)
    ]
    specs += [
        ("PV-B", f"data/prospective/pv-b-300-{r:02d}.ins2D") for r in (1, 2, 3)
    ]
    return specs


def _schedule(name: str, n: int) -> list[tuple[str, int, int, int]]:
    """Predetermined per-instance perturbation list: (op, i, j, length)."""

    rng = random.Random(f"off2b-sched-v1/{name}")
    ops: list[tuple[str, int, int, int]] = []
    for _ in range(N_SWAP):
        ops.append(("swap", rng.randrange(n), rng.randrange(n), 0))
    for _ in range(N_INSERT):
        ops.append(("insert", rng.randrange(n), rng.randrange(n), 0))
    for _ in range(N_REVERSAL):
        length = rng.randint(2, 5)
        start = rng.randrange(n - length + 1) if n > length else 0
        ops.append(("reversal", start, 0, length))
    rng.shuffle(ops)
    assert len(ops) == K_TOTAL
    return ops


def _apply_op(seq: list[Item], op: tuple[str, int, int, int]) -> bool:
    """Mutate ``seq``; return False for recorded no-op (invalid) perturbations."""

    kind, i, j, length = op
    if kind == "swap":
        if i == j:
            return False
        seq[i], seq[j] = seq[j], seq[i]
        return True
    if kind == "insert":
        if i == j:
            return False
        item = seq.pop(i)
        seq.insert(j, item)
        return True
    if length < 2 or i + length > len(seq):
        return False
    seq[i : i + length] = seq[i : i + length][::-1]
    return True


@dataclass(frozen=True, slots=True)
class _Pack:
    height: int
    triplets: frozenset
    coords: Counter
    eval_ms: float
    placements: tuple
    skyline: tuple


def _pack(instance: Instance, config: Config, seq: list[Item]) -> _Pack:
    """Fresh run of ``config`` on ``seq`` via the OFF-2 continuation loop.

    Deliberately NOT ``solver.solve_config``: its widest-fit early break is
    unsound on permuted sequences (sequence-driven walks are unsorted), so
    packing goes through ``off2_state``'s loop with the order-agnostic
    selection. A fresh run is the degenerate continuation of an empty state.
    """

    started = perf_counter()
    placements, skyline, _moves = solve_config_from_state(
        [0] * instance.strip_width, (), seq, config
    )
    eval_ms = (perf_counter() - started) * 1000.0
    ids = {p.item_id for p in placements}
    assert len(ids) == len(instance.items), "completeness broken"
    return _Pack(
        height=max(skyline),
        triplets=frozenset(
            (p.item_id, p.x, p.y, p.width, p.height) for p in placements
        ),
        coords=Counter((p.x, p.y, p.width, p.height) for p in placements),
        eval_ms=eval_ms,
        placements=tuple(placements),
        skyline=tuple(skyline),
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 0.0


def _dispersion(before: _Pack, after: _Pack, strip_width: int) -> tuple[float, int]:
    """Width-normalised mean Manhattan displacement and moved-item count."""

    prev = {p.item_id: p for p in before.placements}
    total = 0
    moved = 0
    for q in after.placements:
        p = prev[q.item_id]
        distance = abs(p.x - q.x) + abs(p.y - q.y)
        total += distance
        if distance or (p.width, p.height) != (q.width, q.height):
            moved += 1
    return total / (len(after.placements) * strip_width), moved


def _walk_cell(
    instance: Instance,
    lb: int,
    bases: dict[str, list[Item]],
    ops: list,
    arm: str,
    group: str,
) -> tuple[list[dict], list[dict]]:
    """One (arm, instance) cell: all four cores walk the same perturbation chain.

    In the main arm all bases are the unified width-descending sequence; in
    the own-ordering arm each core starts from its own default ordering and
    receives the same scheduled operations.
    """

    seqs = {name: list(bases[name]) for name in CORES}
    prev = {
        name: _pack(instance, config, seqs[name]) for name, config in CORES.items()
    }
    for pack in prev.values():
        _validate(instance, list(pack.placements), list(pack.skyline))
    rows: list[dict] = []
    cross: list[dict] = []
    for step, op in enumerate(ops):
        results = [_apply_op(seqs[name], op) for name in CORES]
        valid_op = all(results)
        cur = {
            name: _pack(instance, config, seqs[name])
            for name, config in CORES.items()
        }
        if step % 25 == 24 or step == len(ops) - 1:
            for pack in cur.values():
                _validate(instance, list(pack.placements), list(pack.skyline))
        for name in CORES:
            before, after = prev[name], cur[name]
            delta_h = after.height - before.height
            disp, moved = _dispersion(before, after, instance.strip_width)
            rows.append(
                {
                    "arm": arm,
                    "group": group,
                    "instance": instance.name,
                    "n": len(instance.items),
                    "lb": lb,
                    "step": step,
                    "op": op[0],
                    "pos_i": op[1],
                    "pos_j": op[2],
                    "seg_len": op[3],
                    "valid_op": int(valid_op),
                    "core": name,
                    "h_before": before.height,
                    "h_after": after.height,
                    "delta_h": delta_h,
                    "m": f"{(before.height - after.height) / lb:.6f}",
                    "improved": int(delta_h < 0),
                    "same_height": int(delta_h == 0),
                    "same_layout_phys": int(before.coords == after.coords),
                    "jaccard_id": f"{_jaccard(before.triplets, after.triplets):.6f}",
                    "moved": moved,
                    "mean_disp_norm": f"{disp:.6f}",
                    "eval_ms": f"{after.eval_ms:.3f}",
                }
            )
        for a, b in CROSS_PAIRS:
            cross.append(
                {
                    "arm": arm,
                    "instance": instance.name,
                    "step": step,
                    "pair": f"{a}|{b}",
                    "jaccard": f"{_jaccard(cur[a].triplets, cur[b].triplets):.6f}",
                }
            )
        prev = cur
    return rows, cross


def _bases(items: list[Item], arm: str) -> dict[str, list[Item]]:
    if arm == "main":
        unified = sorted(items, key=_ordering_key(Ordering.WIDTH, rotatable=True))
        return {name: list(unified) for name in CORES}
    return {
        name: sorted(items, key=_ordering_key(config.ordering, rotatable=True))
        for name, config in CORES.items()
    }


def _run(out_dir: str) -> int:
    pert_path = f"{out_dir}/b-perturbations.csv"
    cross_path = f"{out_dir}/b-crosscore.csv"
    done: set[tuple[str, str]] = set()
    if os.path.exists(pert_path):
        counts: Counter = Counter()
        with open(pert_path, newline="") as handle:
            for row in csv.DictReader(handle):
                counts[(row["arm"], row["instance"])] += 1
        done = {key for key, value in counts.items() if value >= len(CORES) * K_TOTAL}
    pert_new = not os.path.exists(pert_path)
    cross_new = not os.path.exists(cross_path)
    started = perf_counter()
    with open(pert_path, "a", newline="") as pert_f, open(
        cross_path, "a", newline=""
    ) as cross_f:
        pert_writer = None
        cross_writer = None
        for group, path in _sample_specs():
            instance = load_instance(path)
            lb = _lower_bound(instance)
            ops = _schedule(instance.name, len(instance.items))
            items = list(instance.items)
            for arm in ARMS:
                if (arm, instance.name) in done:
                    continue
                rows, cross = _walk_cell(
                    instance, lb, _bases(items, arm), ops, arm, group
                )
                if pert_writer is None:
                    pert_writer = csv.DictWriter(pert_f, fieldnames=list(rows[0].keys()))
                    cross_writer = csv.DictWriter(
                        cross_f, fieldnames=list(cross[0].keys())
                    )
                    if pert_new:
                        pert_writer.writeheader()
                    if cross_new:
                        cross_writer.writeheader()
                pert_writer.writerows(rows)
                cross_writer.writerows(cross)
                pert_f.flush()
                cross_f.flush()
                print(
                    f"  {arm:12s} {instance.name:10s} n={len(instance.items):4d} "
                    f"done ({perf_counter() - started:.0f}s)",
                    file=sys.stderr,
                )
    print(f"wrote {pert_path} and {cross_path}", file=sys.stderr)
    return 0


def _parse_states(out_dir: str):
    """Per-(arm, instance, core) step rows, plus initial heights."""

    cells: dict[tuple[str, str, str], list[dict]] = {}
    meta: dict[str, dict] = {}
    with open(f"{out_dir}/b-perturbations.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["arm"], row["instance"], row["core"])
            cells.setdefault(key, []).append(row)
            meta[row["instance"]] = {
                "group": row["group"],
                "n": int(row["n"]),
                "lb": int(row["lb"]),
            }
    for rows in cells.values():
        rows.sort(key=lambda r: int(r["step"]))
    return cells, meta


def _cell_stats(rows: list[dict]) -> dict[str, float]:
    """Per-(arm, instance, core) aggregates over all 200 perturbations."""

    def mean_of(field, pred=lambda r: True):
        vals = [float(r[field]) for r in rows if pred(r)]
        return mean(vals) if vals else 0.0

    stats = {
        "m_swap": mean_of("m", lambda r: r["op"] == "swap"),
        "m_insert": mean_of("m", lambda r: r["op"] == "insert"),
        "m_reversal": mean_of("m", lambda r: r["op"] == "reversal"),
        "improve_rate": mean_of("improved"),
        "same_height_rate": mean_of("same_height"),
        "neutral_rate": mean_of("same_layout_phys"),
        "mean_jaccard": mean_of("jaccard_id"),
        "mean_disp": mean_of("mean_disp_norm"),
        "mean_abs_m": mean([abs(float(r["m"])) for r in rows]),
        "eval_ms": mean_of("eval_ms"),
    }
    stats["initial_h"] = int(rows[0]["h_before"])
    return stats


def _summarize(out_dir: str) -> int:
    cells, meta = _parse_states(out_dir)
    stats = {key: _cell_stats(rows) for key, rows in cells.items()}
    instances = sorted(meta)
    arms = sorted({key[0] for key in cells})
    summary_rows = []

    def add(arm, section, subject, metric, value, ci_lo="", ci_hi="", note=""):
        summary_rows.append(
            {
                "arm": arm,
                "section": section,
                "subject": subject,
                "metric": metric,
                "value": value,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "note": note,
            }
        )

    def paired_ci(values_a, values_b):
        """Instance-paired mean difference with the v2.1 bootstrap."""
        diffs = [a - b for a, b in zip(values_a, values_b)]
        lo, hi = _bootstrap_ci(diffs)
        return mean(diffs), lo, hi

    def mean_ci(values):
        lo, hi = _bootstrap_ci(list(values))
        return mean(values), lo, hi

    trigger = {}
    verdicts = {}
    for arm in arms:
        arm_cells = [key for key in cells if key[0] == arm]
        arm_instances = sorted({key[1] for key in arm_cells})

        # Core x neighbourhood table (m per op type, instance means).
        for core in CORES:
            for op in ("swap", "insert", "reversal"):
                vals = [stats[(arm, name, core)][f"m_{op}"] for name in arm_instances]
                point, lo, hi = mean_ci(vals)
                add(arm, "core_x_neighbourhood", core, f"m_{op}",
                    f"{point:.6f}", f"{lo:.6f}", f"{hi:.6f}")
            for metric in ("improve_rate", "neutral_rate", "same_height_rate",
                           "mean_jaccard", "mean_disp", "mean_abs_m", "eval_ms"):
                vals = [stats[(arm, name, core)][metric] for name in arm_instances]
                point, lo, hi = mean_ci(vals)
                add(arm, "core_stats", core, metric, f"{point:.6f}", f"{lo:.6f}", f"{hi:.6f}")

        # B1: paired core-pair contrasts on response statistics.
        b1_support = False
        for a, b in CROSS_PAIRS:
            for metric in ("improve_rate", "neutral_rate", "same_height_rate",
                           "mean_jaccard", "mean_abs_m"):
                point, lo, hi = paired_ci(
                    [stats[(arm, name, a)][metric] for name in arm_instances],
                    [stats[(arm, name, b)][metric] for name in arm_instances],
                )
                support = lo > 0 or hi < 0
                b1_support |= support
                add(arm, "b1_contrast", f"{a}-{b}", metric,
                    f"{point:+.6f}", f"{lo:+.6f}", f"{hi:+.6f}",
                    note="CI excludes 0" if support else "")

        # B2: pre-specified difference-in-differences I (S2 vs BF, insert vs swap).
        def m(core, op, name):
            return stats[(arm, name, core)][f"m_{op}"]

        inter = {}
        for a, b in (("S2", "BF"), ("S1", "BF"), ("S2-TRoff", "BF")):
            diffs = [
                (m(a, "insert", name) - m(a, "swap", name))
                - (m(b, "insert", name) - m(b, "swap", name))
                for name in arm_instances
            ]
            point, lo, hi = mean_ci(diffs)
            inter[(a, b)] = (point, lo, hi)
            add(arm, "b2_interaction", f"I({a}-{b})", "insert-vs-swap",
                f"{point:+.6f}", f"{lo:+.6f}", f"{hi:+.6f}",
                note="pre-specified" if (a, b) == ("S2", "BF") else "secondary")
        # Tower-removal contribution: S2 minus S2-TRoff per neighbourhood.
        for op in ("swap", "insert", "reversal"):
            point, lo, hi = paired_ci(
                [m("S2", op, name) for name in arm_instances],
                [m("S2-TRoff", op, name) for name in arm_instances],
            )
            add(arm, "tr_contribution", "S2-(S2-TRoff)", op,
                f"{point:+.6f}", f"{lo:+.6f}", f"{hi:+.6f}")

        # Initial-gap-close subset (<= 1% LB span of the four initial heights).
        close = []
        for name in arm_instances:
            heights = [stats[(arm, name, core)]["initial_h"] for core in CORES]
            span = (max(heights) - min(heights)) / meta[name]["lb"]
            if span <= 0.01:
                close.append(name)
        if len(close) >= 6:
            diffs = [
                (m("S2", "insert", name) - m("S2", "swap", name))
                - (m("BF", "insert", name) - m("BF", "swap", name))
                for name in close
            ]
            point, lo, hi = mean_ci(diffs)
            add(arm, "subset_initial_gap_close", "I(S2-BF)", "insert-vs-swap",
                f"{point:+.6f}", f"{lo:+.6f}", f"{hi:+.6f}",
                note=f"{len(close)} instances with initial-gap span <= 1% LB")
        else:
            add(arm, "subset_initial_gap_close", "I(S2-BF)", "insert-vs-swap", "NA",
                note=f"only {len(close)} instances with span <= 1% LB (< 6)")

        # Cross-core after-layout divergence (descriptive).
        cross_path = f"{out_dir}/b-crosscore.csv"
        pair_vals: dict[tuple[str, str], list[float]] = {}
        with open(cross_path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["arm"] != arm or row["instance"] not in arm_instances:
                    continue
                pair_vals.setdefault((row["pair"], row["instance"]), []).append(
                    float(row["jaccard"])
                )
        for pair in sorted({p for p, _i in pair_vals}):
            vals = [mean(pair_vals[(pair, name)]) for name in arm_instances]
            point, lo, hi = mean_ci(vals)
            add(arm, "cross_core_divergence", pair, "mean_jaccard_after",
                f"{point:.4f}", f"{lo:.4f}", f"{hi:.4f}")

        i_point, i_lo, i_hi = inter[("S2", "BF")]
        b2_support = i_lo > 0 or i_hi < 0
        triggered = (b1_support or b2_support) and b2_support
        trigger[arm] = triggered
        verdicts[arm] = {
            "B1": "supported-evidence" if b1_support else "insufficient",
            "B2": "supported-evidence" if b2_support else "insufficient",
            "B3": "triggered" if triggered else "not-run-not-evaluated",
        }
        add(arm, "trigger", "B1", "support", int(b1_support))
        add(arm, "trigger", "B2", "support (I CI excludes 0)", int(b2_support))
        add(arm, "trigger", "dynamic", "triggered", int(triggered),
            note="B.4: dynamic arm runs only if B1 or B2 supported AND I's CI excludes 0")

    with open(f"{out_dir}/b-summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for arm in arms:
        print(f"\n===== arm={arm} =====")
        print("core stats (instance means):")
        header = f"{'core':10s} {'m_swap':>8s} {'m_insert':>9s} {'m_reversal':>10s} {'impr':>6s} {'neutral':>8s} {'sameH':>6s} {'jac':>6s} {'eval_ms':>8s}"
        print(header)
        for core in CORES:
            s = [r for r in summary_rows if r["arm"] == arm and r["subject"] == core]
            g = lambda metric: next(r["value"] for r in s if r["metric"] == metric)
            print(f"{core:10s} {float(g('m_swap')):8.4f} {float(g('m_insert')):9.4f} "
                  f"{float(g('m_reversal')):10.4f} {float(g('improve_rate')):6.3f} "
                  f"{float(g('neutral_rate')):8.3f} {float(g('same_height_rate')):6.3f} "
                  f"{float(g('mean_jaccard')):6.3f} {float(g('eval_ms')):8.2f}")
        print("\nB2 interaction I = (m_insert - m_swap) difference-in-differences:")
        for r in summary_rows:
            if r["arm"] == arm and r["section"] == "b2_interaction":
                print(f"  {r['subject']:16s} {r['value']:>9s} [{r['ci_lo']:>9s}, {r['ci_hi']:>9s}]  {r['note']}")
        print("B1 contrasts with CI excluding 0:")
        any_b1 = False
        for r in summary_rows:
            if r["arm"] == arm and r["section"] == "b1_contrast" and r["note"]:
                print(f"  {r['subject']:16s} {r['metric']:16s} {r['value']:>9s} [{r['ci_lo']:>9s}, {r['ci_hi']:>9s}]")
                any_b1 = True
        if not any_b1:
            print("  (none)")
        for r in summary_rows:
            if r["arm"] == arm and r["section"] == "subset_initial_gap_close":
                print(f"subset (initial-gap<=1%LB): {r['value']} [{r['ci_lo']}, {r['ci_hi']}]  {r['note']}")
        v = verdicts[arm]
        print(f"verdicts: B1={v['B1']}  B2={v['B2']}  dynamic B3={v['B3']}")
    print(f"\nwrote {out_dir}/b-summary.csv", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cch.off2_landscape",
        description=__doc__.splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "summarize"):
        child = sub.add_parser(name)
        child.add_argument("--out-dir", default="docs/off2-learn")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.command == "run":
        return _run(args.out_dir)
    return _summarize(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
