"""OFF-2 validation A (lookahead gate) and the shared engineering acceptances.

Implements the frozen protocols of the companion design documents (dated, in the data package)
and the engineering preconditions of ``design-b.md`` (B.2):

    python3 -m cch.off2_validation accept-state     # A.3: resume == one-shot
    python3 -m cch.off2_validation accept-sequence  # B.2: explicit sequence is not reordered
    python3 -m cch.off2_validation run              # A main experiment -> a-states.csv
    python3 -m cch.off2_validation summarize        # -> a-regret.csv + a-summary.csv

Protocol operationalisations fixed by this module (2026-09-17, within the
frozen text; no protocol change):

- Checkpoints: the state after placement number ``ceil(p * n)`` for
  p in {25%, 50%, 75%} of the reference run (width/widest-fit/LM, no tower
  removal, no vertical niche).
- Reference-kernel rollouts (``_roll3``) mirror the engine's plain
  lowest-gap/widest-fit/LM path, limited to three placements.
- "Newly buried area" of a candidate (an immediate quantity named but not
  formulated in the design): zero when the residual slit is closed, fillable
  by some remaining item (``min(w, h) <= r``, matching the rotatable engine),
  or nothing remains; otherwise the area the first raise over the unfillable
  slit would bury, ``r * (min(walls) - gap.y)``.
- P_comp2 orientations: both normalised orientations are legal (v2.1 text),
  so each item contributes extents {w, h}; pairs minimise r - (e_i + e_j)
  over feasible orientation combos. With >= 2 items and no feasible pair the
  single-item fallback applies ("no feasible single or pair -> +inf");
  flags record the branch taken.
- Random candidates: uniform sample without replacement (seed per
  instance/checkpoint, ``off2a-v1/<name>/<pct>``) from all corner placements
  of the current lowest gap (both orientations x both sides, deduplicated).
- Bootstrap: 10,000 instance-level paired resamples, seed 20260917,
  percentile interval.
- This module drives the engine's ``_select`` on width-sorted lists only
  (reference kernel = width ordering; all continuations/rollouts consume
  sorted subsequences), so the engine's widest-fit early break is sound
  here — verified 2026-09-17 by an instrumented rerun (24,076 calls, all
  lists non-increasing) reproducing a-states.csv byte-identically. Any
  future unsorted-sequence use must route through
  ``cch/off2_state._select_any_order`` instead.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from math import ceil, inf, sqrt
from statistics import mean, median

from burke_bf import (
    Instance,
    Item,
    Placement,
    Policy,
    Solution,
    load_instance,
    validate_solution,
)

from .experiment import _lower_bound
from .model import Config, Ordering, PlacementPolicy, Selection
from .off2_state import solve_config_from_state, solve_config_traced
from .solver import (
    _lowest_gap,
    _neighbours,
    _place_on_left,
    _raise,
    _Rectangle,
    _select,
    solve_config,
)

# Reference kernel pi: classic BF (width / widest-fit / LM), no TR, no VN.
REF_CONFIG = Config(
    ordering=Ordering.WIDTH,
    selection=Selection.WIDEST_FIT,
    placement=PlacementPolicy.LM,
)
CHECKPOINT_PCTS = (25, 50, 75)
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260917

SCORERS = ("dh", "ref", "g0", "p_roll3", "p_fill_area", "p_comp2")
BASELINES = ("dh", "ref", "g0")
PROXIES = ("p_roll3", "p_fill_area", "p_comp2")
MAIN_CONTRAST = ("p_roll3", "g0")


def _instance_specs() -> list[tuple[str, str]]:
    """The 36 development instances of design A.1: (group, path) pairs."""

    specs = [("C", f"data/c/c{c}-p{p}.ins2D") for c in (1, 2, 3) for p in (1, 2, 3)]
    specs += [("NT", f"data/nt/N{c}{s}.ins2D") for c in (1, 2, 3) for s in "abcde"]
    specs += [
        (f"PV-{g}", f"data/prospective/pv-{g}-100-{r:02d}.ins2D")
        for g in "abcd"
        for r in (1, 2, 3)
    ]
    return specs


def _rects(items: list[Item] | tuple[Item, ...], rotatable: bool) -> list[_Rectangle]:
    if rotatable:
        return [
            _Rectangle(item, max(item.width, item.height), min(item.width, item.height))
            for item in items
        ]
    return [_Rectangle(item, item.width, item.height) for item in items]


def _preraise(skyline: list[int], remaining: list[_Rectangle], config: Config):
    """Deterministic gap raising until the lowest gap accepts a rectangle."""

    raises = 0
    while True:
        gap = _lowest_gap(skyline)
        if _select(config, remaining, skyline, gap) is not None:
            return gap, raises
        _raise(skyline, gap)
        raises += 1


@dataclass(frozen=True, slots=True)
class _Cand:
    kind: str  # ref | opposite | second | random
    item: Item
    w: int
    h: int
    x: int
    y: int

    @property
    def code(self) -> tuple:
        return (self.item.item_id, self.w, self.h, self.x, self.y)


def _candidates(
    skyline: list[int],
    remaining: list[_Rectangle],
    gap,
    config: Config,
    rng: random.Random,
) -> list[_Cand]:
    """The fixed candidate set of design A.1 at the current lowest gap."""

    cands: list[_Cand] = []
    seen: set[tuple] = set()

    def add(item: Item, w: int, h: int, x: int, kind: str) -> None:
        key = (item.item_id, w, h, x, gap.y)
        if key not in seen:
            seen.add(key)
            cands.append(_Cand(kind, item, w, h, x, gap.y))

    index, rect, w, h = _select(config, remaining, skyline, gap)
    on_left = _place_on_left(config.placement, skyline, gap, gap.y + h)
    add(rect.item, w, h, gap.x if on_left else gap.right - w, "ref")
    add(rect.item, w, h, gap.right - w if on_left else gap.x, "opposite")
    rest = remaining[:index] + remaining[index + 1 :]
    found = _select(config, rest, skyline, gap)
    if found is not None:
        _i2, rect2, w2, h2 = found
        on_left2 = _place_on_left(config.placement, skyline, gap, gap.y + h2)
        add(rect2.item, w2, h2, gap.x if on_left2 else gap.right - w2, "second")
    feasible: list[tuple] = []
    feas_seen: set[tuple] = set()
    for r in remaining:
        orientations = [(r.width, r.height)]
        if config.rotatable and r.width != r.height:
            orientations.append((r.height, r.width))
        for fw, fh in orientations:
            if fw > gap.width:
                continue
            for side_left in (True, False):
                fx = gap.x if side_left else gap.right - fw
                fkey = (r.item.item_id, fw, fh, fx, gap.y)
                if fkey not in feas_seen:
                    feas_seen.add(fkey)
                    feasible.append((r.item, fw, fh, fx))
    for item, fw, fh, fx in rng.sample(feasible, k=min(2, len(feasible))):
        add(item, fw, fh, fx, "random")
    return cands


def _fits_slit(item: Item, residual: int, rotatable: bool) -> bool:
    if rotatable:
        return min(item.width, item.height) <= residual
    return item.width <= residual


def _new_buried(
    skyline: list[int], gap, cand: _Cand, remaining_items: list[Item], rotatable: bool
) -> int:
    """First-raise burial estimate of the residual slit (module docstring)."""

    residual = gap.width - cand.w
    if residual == 0 or not remaining_items:
        return 0
    if any(_fits_slit(item, residual, rotatable) for item in remaining_items):
        return 0
    left, right = _neighbours(skyline, gap)
    placed_left = cand.x == gap.x
    wall = min(cand.y + cand.h, right if placed_left else left)
    return residual * int(wall - gap.y)


def _p_fill_area(remaining_items: list[Item], residual: int, rotatable: bool) -> float:
    """Area share of remaining items that could still enter the slit (big=good)."""

    if residual == 0:
        return 1.0  # closed slit: no future loss, recorded as the best value
    total = sum(item.area for item in remaining_items)
    if total == 0:
        return 1.0
    fillable = sum(
        item.area for item in remaining_items if _fits_slit(item, residual, rotatable)
    )
    return fillable / total


def _p_comp2(remaining_items: list[Item], residual: int) -> tuple[float, str]:
    """Two-piece complement proximity of the slit width (small=good)."""

    if residual == 0:
        return 0.0, "closed"
    if not remaining_items:
        return 0.0, "empty"
    extents = [sorted({item.width, item.height}) for item in remaining_items]
    if len(extents) == 1:
        feasible = [e for e in extents[0] if e <= residual]
        if not feasible:
            return inf, "infeasible-single"
        return float(residual - max(feasible)), "single"
    best = None
    for i in range(len(extents)):
        for j in range(i + 1, len(extents)):
            sums = [a + b for a in extents[i] for b in extents[j] if a + b <= residual]
            if sums:
                value = residual - max(sums)
                best = value if best is None else min(best, value)
    if best is not None:
        return float(best), "pair"
    singles = [e for extent in extents for e in extent if e <= residual]
    if singles:
        return float(residual - max(singles)), "single-fallback"
    return inf, "infeasible"


def _roll3(
    skyline: list[int], remaining: list[_Rectangle], config: Config
) -> tuple[int, int, int, int]:
    """Reference-kernel rollout capped at 3 placements: (height, steps, buried, area)."""

    skyline = list(skyline)
    remaining = list(remaining)
    steps = 0
    buried = 0
    area = 0
    while remaining and steps < 3:
        gap = _lowest_gap(skyline)
        found = _select(config, remaining, skyline, gap)
        if found is None:
            left, right = _neighbours(skyline, gap)
            buried += gap.width * int(min(left, right) - gap.y)
            _raise(skyline, gap)
            continue
        index, rect, w, h = found
        on_left = _place_on_left(config.placement, skyline, gap, gap.y + h)
        x = gap.x if on_left else gap.right - w
        skyline[x : x + w] = [gap.y + h] * w
        area += rect.item.area
        del remaining[index]
        steps += 1
    return max(skyline), steps, buried, area


def _validate(instance: Instance, placements: list[Placement], skyline: list[int]) -> None:
    solution = Solution(
        instance_name=instance.name,
        policy=Policy.LEFTMOST,
        height=max(skyline),
        initial_height=max(skyline),
        placements=tuple(placements),
        skyline=tuple(skyline),
        tower_moves=0,
    )
    validate_solution(instance, solution)


def _evaluate_state(
    instance: Instance, lb: int, state, pct: int
) -> tuple[list[dict], dict]:
    """Pre-raise, generate candidates, score them, and continue each to completion."""

    config = REF_CONFIG
    skyline = list(state.skyline)
    placements = list(state.placements)
    remaining = _rects(state.remaining, config.rotatable)
    gap, raises = _preraise(skyline, remaining, config)
    h_t = max(skyline)
    left, right = _neighbours(skyline, gap)
    lh = left - gap.y if left != inf else inf
    rh = right - gap.y if right != inf else inf
    rng = random.Random(f"off2a-v1/{instance.name}/{pct}")
    cands = _candidates(skyline, remaining, gap, config, rng)
    rows = []
    for ix, cand in enumerate(cands):
        remaining_rects = [r for r in remaining if r.item.item_id != cand.item.item_id]
        remaining_items = [r.item for r in remaining_rects]
        delta_h = max(0, cand.y + cand.h - h_t)
        residual = gap.width - cand.w
        fitness = (cand.w == gap.width) + (cand.h == lh) + (cand.h == rh)
        buried = _new_buried(skyline, gap, cand, remaining_items, config.rotatable)
        p_fill = _p_fill_area(remaining_items, residual, config.rotatable)
        p_comp, p_comp_flag = _p_comp2(remaining_items, residual)
        sky_forced = list(skyline)
        sky_forced[cand.x : cand.x + cand.w] = [cand.y + cand.h] * cand.w
        h_t3, steps, roll_buried, roll_area = _roll3(sky_forced, remaining_rects, config)
        delta_a = cand.item.area + roll_area
        delta_bt = instance.strip_width * (h_t3 - h_t) - delta_a
        placements_forced = placements + [
            Placement(
                item_id=cand.item.item_id,
                x=cand.x,
                y=cand.y,
                width=cand.w,
                height=cand.h,
                original_width=cand.item.width,
                original_height=cand.item.height,
            )
        ]
        cont_p, cont_s, _ = solve_config_from_state(
            sky_forced, placements_forced, remaining_items, config
        )
        _validate(instance, cont_p, cont_s)
        rows.append(
            {
                "cand_ix": ix,
                "kind": cand.kind,
                "item_id": cand.item.item_id,
                "w": cand.w,
                "h": cand.h,
                "x": cand.x,
                "y": cand.y,
                "delta_h": delta_h,
                "residual": residual,
                "new_buried": buried,
                "fitness": fitness,
                "p_fill_area": p_fill,
                "p_comp2": p_comp,
                "p_comp2_flag": p_comp_flag,
                "roll_h": h_t3,
                "roll_steps": steps,
                "roll_da": delta_a,
                "roll_db": roll_buried,
                "roll_dbt": delta_bt,
                "q_pi": max(cont_s),
            }
        )
    meta = {"raises": raises, "h_t": h_t, "gap": gap}
    return rows, meta


def _run(out_dir: str) -> int:
    states_path = f"{out_dir}/a-states.csv"
    fields = [
        "instance", "group", "n", "lb", "checkpoint_pct", "placed_k", "placed_frac",
        "preraise_raises", "h_t", "n_cand", "excluded", "decision_space", "q_min",
        "cand_ix", "kind", "item_id", "w", "h", "x", "y",
        "delta_h", "residual", "new_buried", "fitness",
        "p_fill_area", "p_comp2", "p_comp2_flag",
        "roll_h", "roll_steps", "roll_da", "roll_db", "roll_dbt", "q_pi",
    ]
    n_states = 0
    with open(states_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, path in _instance_specs():
            instance = load_instance(path)
            lb = _lower_bound(instance)
            n = len(instance.items)
            _p, _s, _m, trace = solve_config_traced(instance, REF_CONFIG)
            assert len(trace) == n
            for pct in CHECKPOINT_PCTS:
                k = ceil(pct * n / 100)
                state = trace[k - 1]
                rows, meta = _evaluate_state(instance, lb, state, pct)
                q_min = min(row["q_pi"] for row in rows)
                decision_space = int(
                    len({row["q_pi"] for row in rows}) > 1
                )
                excluded = int(len(rows) < 2)
                for row in rows:
                    writer.writerow(
                        {
                            "instance": instance.name,
                            "group": group,
                            "n": n,
                            "lb": lb,
                            "checkpoint_pct": pct,
                            "placed_k": k,
                            "placed_frac": f"{k / n:.4f}",
                            "preraise_raises": meta["raises"],
                            "h_t": meta["h_t"],
                            "n_cand": len(rows),
                            "excluded": excluded,
                            "decision_space": decision_space,
                            "q_min": q_min,
                            **{k2: v for k2, v in row.items()},
                        }
                    )
                n_states += 1
            print(f"  {instance.name:10s} n={n:4d} states done", file=sys.stderr)
    print(f"wrote {states_path}: {n_states} states", file=sys.stderr)
    return 0


def _choose(cands: list[dict], scorer: str) -> dict:
    """Scorer's action; ties keep the reference action, then smallest action code."""

    if scorer == "ref":
        return next(c for c in cands if c["kind"] == "ref")
    if scorer == "dh":
        key, maximize = lambda c: c["delta_h"], False
    elif scorer == "g0":
        key, maximize = lambda c: (-c["fitness"], c["delta_h"], c["residual"]), False
    elif scorer == "p_roll3":
        key, maximize = lambda c: c["roll_h"], False
    elif scorer == "p_fill_area":
        key, maximize = lambda c: c["p_fill_area"], True
    elif scorer == "p_comp2":
        key, maximize = lambda c: c["p_comp2"], False
    else:
        raise AssertionError(f"unknown scorer {scorer}")
    scores = [key(c) for c in cands]
    best = max(scores) if maximize else min(scores)
    tied = [c for c in cands if key(c) == best]
    refs = [c for c in tied if c["kind"] == "ref"]
    if refs:
        return refs[0]
    return min(tied, key=lambda c: (c["item_id"], c["w"], c["h"], c["x"], c["y"]))


def _badness(cand: dict, scorer: str):
    """Scalar (or lexicographic tuple) where larger means worse, for Kendall tau."""

    if scorer == "dh":
        return cand["delta_h"]
    if scorer == "g0":
        return (-cand["fitness"], cand["delta_h"], cand["residual"])
    if scorer == "p_roll3":
        return cand["roll_h"]
    if scorer == "p_fill_area":
        return -cand["p_fill_area"]
    if scorer == "p_comp2":
        return cand["p_comp2"]
    raise AssertionError(f"no score for {scorer}")


def _tau_b(xs: list, ys: list) -> float | None:
    """Kendall tau-b; None (NA) when one side is entirely tied."""

    concordant = discordant = tie_x = tie_y = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None


def _bootstrap_ci(instance_means: list[float]) -> tuple[float, float]:
    """Instance-level paired bootstrap percentile interval (fixed seed)."""

    rng = random.Random(BOOTSTRAP_SEED)
    n = len(instance_means)
    estimates = sorted(
        mean(instance_means[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_REPS)
    )
    return estimates[int(0.025 * BOOTSTRAP_REPS)], estimates[int(0.975 * BOOTSTRAP_REPS) - 1]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _summarize(out_dir: str) -> int:
    states: dict[tuple[str, int], dict] = {}
    with open(f"{out_dir}/a-states.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["instance"], int(row["checkpoint_pct"]))
            state = states.setdefault(
                key,
                {
                    "instance": row["instance"],
                    "group": row["group"],
                    "lb": int(row["lb"]),
                    "pct": int(row["checkpoint_pct"]),
                    "excluded": int(row["excluded"]),
                    "decision_space": int(row["decision_space"]),
                    "q_min": int(row["q_min"]),
                    "cands": [],
                },
            )
            state["cands"].append(
                {
                    "cand_ix": int(row["cand_ix"]),
                    "kind": row["kind"],
                    "item_id": row["item_id"],
                    "w": int(row["w"]),
                    "h": int(row["h"]),
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "delta_h": int(row["delta_h"]),
                    "residual": int(row["residual"]),
                    "new_buried": int(row["new_buried"]),
                    "fitness": int(row["fitness"]),
                    "p_fill_area": float(row["p_fill_area"]),
                    "p_comp2": float(row["p_comp2"]),
                    "roll_h": int(row["roll_h"]),
                    "roll_steps": int(row["roll_steps"]),
                    "roll_da": int(row["roll_da"]),
                    "roll_db": int(row["roll_db"]),
                    "roll_dbt": int(row["roll_dbt"]),
                    "q_pi": int(row["q_pi"]),
                }
            )
    evaluated = [s for s in states.values() if not s["excluded"]]
    regret_rows = []
    regrets: dict[str, dict[tuple[str, int], float]] = {f: {} for f in SCORERS}
    for state in evaluated:
        lb, q_min = state["lb"], state["q_min"]
        for scorer in SCORERS:
            chosen = _choose(state["cands"], scorer)
            regret = (chosen["q_pi"] - q_min) / lb
            regrets[scorer][(state["instance"], state["pct"])] = regret
            regret_rows.append(
                {
                    "instance": state["instance"],
                    "group": state["group"],
                    "checkpoint_pct": state["pct"],
                    "scorer": scorer,
                    "cand_ix": chosen["cand_ix"],
                    "q_chosen": chosen["q_pi"],
                    "q_min": q_min,
                    "regret": f"{regret:.6f}",
                }
            )
    with open(f"{out_dir}/a-regret.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regret_rows[0].keys()))
        writer.writeheader()
        writer.writerows(regret_rows)

    summary_rows = []

    def add(section, subject, versus, metric, value, ci_lo="", ci_hi="", note=""):
        summary_rows.append(
            {
                "section": section,
                "subject": subject,
                "versus": versus,
                "metric": metric,
                "value": value,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_states": len(evaluated),
                "n_instances": len({s["instance"] for s in evaluated}),
                "note": note,
            }
        )

    # Per-scorer regret distribution (baselines double as the improvement
    # ceiling distribution of design A.4).
    for scorer in SCORERS:
        values = list(regrets[scorer].values())
        add("regret", scorer, "", "mean", f"{mean(values):.6f}")
        add("regret", scorer, "", "p50", f"{median(values):.6f}")
        add("regret", scorer, "", "p90", f"{_quantile(values, 0.9):.6f}")
        add("regret", scorer, "", "max", f"{max(values):.6f}")
        add("regret", scorer, "", "frac_positive", f"{mean(v > 0 for v in values):.4f}",
            note="baseline rows = improvement-ceiling distribution" if scorer in BASELINES else "")

    # Paired comparisons with instance-level bootstrap intervals.
    instance_names = sorted({s["instance"] for s in evaluated})
    cis: dict[tuple[str, str], tuple[float, float, float]] = {}
    for proxy in PROXIES:
        for baseline in BASELINES:
            per_instance = []
            for name in instance_names:
                diffs = [
                    regrets[baseline][(name, s["pct"])] - regrets[proxy][(name, s["pct"])]
                    for s in evaluated
                    if s["instance"] == name
                ]
                per_instance.append(mean(diffs))
            point = mean(per_instance)
            lo, hi = _bootstrap_ci(per_instance)
            cis[(proxy, baseline)] = (point, lo, hi)
            add(
                "comparison", proxy, baseline, "mean_d",
                f"{point:.6f}", f"{lo:.6f}", f"{hi:.6f}",
                note="d = r(baseline) - r(proxy); positive favours proxy",
            )

    # Kendall tau-b between proxy scores and completion height.
    for scorer in ("dh", "g0") + PROXIES:
        taus, na = [], 0
        for state in evaluated:
            tau = _tau_b(
                [_badness(c, scorer) for c in state["cands"]],
                [c["q_pi"] for c in state["cands"]],
            )
            if tau is None:
                na += 1
            else:
                taus.append(tau)
        add("tau", scorer, "", "mean_tau_b", f"{mean(taus):.4f}" if taus else "NA",
            note=f"na={na} all-tied states")

    # Decision-space proportion and single-candidate exclusions.
    n_all = len(states)
    add("decision_space", "", "", "proportion",
        f"{mean(s['decision_space'] for s in evaluated):.4f}",
        note=f"evaluated={len(evaluated)} of {n_all} states; excluded single-candidate={n_all - len(evaluated)}")

    # P_roll3 exploratory: the four roll quantities and their within-state
    # association with Q_pi (area-confound check, design A.2).
    for metric in ("roll_h", "roll_da", "roll_db", "roll_dbt"):
        values = [c[metric] for s in evaluated for c in s["cands"]]
        add("roll3", "p_roll3", "", f"mean_{metric}", f"{mean(values):.4f}")
    for metric in ("roll_h", "roll_da", "roll_db", "roll_dbt"):
        xs, ys = [], []
        for state in evaluated:
            if not state["decision_space"]:
                continue
            cx = [c[metric] for c in state["cands"]]
            cy = [c["q_pi"] for c in state["cands"]]
            xs.extend(x - mean(cx) for x in cx)
            ys.extend(y - mean(cy) for y in cy)
        corr = _pearson(xs, ys) if xs else None
        add("roll3", "p_roll3", "", f"within_corr_{metric}_q",
            f"{corr:.4f}" if corr is not None else "NA",
            note="within-state centred Pearson, decision-space states")

    # Verdict under the frozen decision rule (design A.4: three tiers only).
    main_point, main_lo, main_hi = cis[MAIN_CONTRAST]
    all_lo = [cis[("p_roll3", b)][1] for b in BASELINES]
    if all(lo > 0 for lo in all_lo):
        verdict, note = "support", "all three baseline contrasts have CI lower bound > 0"
    elif main_lo > 0:
        verdict, note = "partial", "main contrast (G0) favoured; some auxiliary contrast uncertain"
    else:
        verdict, note = "no-increment", (
            "no incremental value sufficient to justify further investment was "
            "detected under the tested state distribution, candidate sets, and "
            "pre-registered proxies (per A.0 wording; not a refutation of "
            "lookahead construction in general)"
        )
    add("judgment", "p_roll3", "g0", "verdict", verdict, note=note)

    with open(f"{out_dir}/a-summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"evaluated states: {len(evaluated)} / {len(states)} "
          f"({len(instance_names)} instances)")
    print("\n== per-scorer regret (LB-normalised) ==")
    print(f"{'scorer':12s} {'mean':>8s} {'p50':>8s} {'p90':>8s} {'max':>8s} {'frac>0':>7s}")
    for scorer in SCORERS:
        values = list(regrets[scorer].values())
        print(f"{scorer:12s} {mean(values):8.4f} {median(values):8.4f} "
              f"{_quantile(values, 0.9):8.4f} {max(values):8.4f} "
              f"{mean(v > 0 for v in values):7.3f}")
    print("\n== three-baseline contrasts d = r(baseline) - r(proxy) ==")
    print(f"{'proxy':12s} {'baseline':8s} {'mean_d':>8s} {'95% CI':>20s}")
    for proxy in PROXIES:
        for baseline in BASELINES:
            point, lo, hi = cis[(proxy, baseline)]
            print(f"{proxy:12s} {baseline:8s} {point:8.4f} [{lo:8.4f}, {hi:8.4f}]")
    print("\n== Kendall tau-b (proxy vs Q_pi) ==")
    for row in summary_rows:
        if row["section"] == "tau":
            print(f"{row['subject']:12s} {row['value']:>7s}  {row['note']}")
    print(f"\njudgment: {verdict} — {note}")
    print(f"\nwrote {out_dir}/a-regret.csv and {out_dir}/a-summary.csv", file=sys.stderr)
    return 0


def _accept_state(out_dir: str) -> int:
    """A.3 acceptance: truncated resume vs one-shot run, plus deep-copy checks."""

    rows = []
    total_states = 0
    for group, path in _instance_specs():
        instance = load_instance(path)
        full_p, full_s, full_m = solve_config(instance, REF_CONFIG)
        trc_p, trc_s, trc_m, trace = solve_config_traced(instance, REF_CONFIG)
        sane = trc_p == full_p and trc_s == full_s and trc_m == full_m
        _validate(instance, full_p, full_s)
        n = len(instance.items)
        assert len(trace) == n
        failures = []
        for k in range(1, n):
            state = trace[k - 1]
            cont_p, cont_s, _m = solve_config_from_state(
                state.skyline, state.placements, state.remaining, REF_CONFIG
            )
            trajectory = cont_p == full_p and cont_p[k:] == full_p[k:]
            orientations = {p.item_id: (p.width, p.height) for p in cont_p} == {
                p.item_id: (p.width, p.height) for p in full_p
            }
            layout = (
                {p.item_id: (p.x, p.y, p.width, p.height) for p in cont_p}
                == {p.item_id: (p.x, p.y, p.width, p.height) for p in full_p}
                and cont_s == full_s
                and max(cont_s) == max(full_s)
            )
            try:
                _validate(instance, cont_p, cont_s)
                valid = True
            except ValueError:
                valid = False
            if not (trajectory and orientations and layout and valid):
                failures.append((k, trajectory, orientations, layout, valid))
            total_states += 1
        # Deep-copy verification at the 50% checkpoint: inputs must survive a
        # continuation untouched, and two runs from one snapshot must match.
        k = ceil(50 * n / 100)
        state = trace[k - 1]
        sky_in = list(state.skyline)
        pla_in = list(state.placements)
        rem_in = list(state.remaining)
        r1_p, r1_s, _ = solve_config_from_state(sky_in, pla_in, rem_in, REF_CONFIG)
        r1_p.pop()  # mutate the first run's outputs...
        r1_s[0] = -1
        inputs_intact = (
            sky_in == list(state.skyline)
            and pla_in == list(state.placements)
            and rem_in == list(state.remaining)
        )
        r2_p, r2_s, _ = solve_config_from_state(
            state.skyline, state.placements, state.remaining, REF_CONFIG
        )
        deep_ok = inputs_intact and r2_p == full_p and r2_s == full_s
        rows.append(
            {
                "instance": instance.name,
                "group": group,
                "n": n,
                "states_checked": n - 1,
                "traced_eq_oneshot": int(sane),
                "failures": len(failures),
                "deepcopy_ok": int(deep_ok),
                "first_failure": failures[0] if failures else "",
            }
        )
        status = "OK" if sane and not failures and deep_ok else "FAIL"
        print(f"  {instance.name:10s} states={n - 1:4d} {status}", file=sys.stderr)
    with open(f"{out_dir}/accept-state.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    bad = [r for r in rows if r["failures"] or not r["traced_eq_oneshot"] or not r["deepcopy_ok"]]
    print(
        f"accept-state: {len(rows) - len(bad)}/{len(rows)} instances pass, "
        f"{total_states} intermediate states, four-way equality "
        "(trajectory/orientation/layout/validity) + deep-copy"
    )
    if bad:
        print(f"FAILURES: {[r['instance'] for r in bad]}", file=sys.stderr)
        return 1
    return 0


def _probe_instance() -> Instance:
    """All-full-width probe: the first remaining rectangle always fits the
    current gap exactly, so placement order equals the input list order."""

    return Instance(
        name="SEQPROBE",
        strip_width=10,
        reference_height=None,
        items=tuple(Item(f"i{h:02d}", 10, h) for h in range(1, 13)),
    )


def _ordering_key(ordering: Ordering, rotatable: bool):
    """Item-level replica of ``solver._sort`` keys (to build anchor sequences)."""

    def dims(item: Item) -> tuple[int, int]:
        if rotatable:
            return max(item.width, item.height), min(item.width, item.height)
        return item.width, item.height

    keys = {
        Ordering.WIDTH: lambda it: (-dims(it)[0], -dims(it)[1]),
        Ordering.HEIGHT: lambda it: (-dims(it)[1], -dims(it)[0]),
        Ordering.AREA: lambda it: (-dims(it)[0] * dims(it)[1], -dims(it)[0]),
        Ordering.PERIMETER: lambda it: (-2 * sum(dims(it)), -dims(it)[0]),
        Ordering.MAXSIDE: lambda it: (-max(dims(it)), -dims(it)[1]),
        Ordering.DIAGONAL: lambda it: (
            -(sqrt(dims(it)[0] ** 2 + dims(it)[1] ** 2) + sum(dims(it))),
            -dims(it)[0],
        ),
    }
    return keys[ordering]


def _accept_sequence(out_dir: str) -> int:
    """B.2 precondition: an explicit sequence is packed as given, not reordered.

    Placement order is decided by the selection rule, so the assertions are
    behavioural: (probe) with all-full-width items the placement order equals
    the input sequence exactly, under every configured ordering; (anchor)
    feeding an ordering's sorted list as the explicit sequence reproduces the
    sequence-less run exactly; (control) sequence-less WIDTH vs HEIGHT runs
    differ on real instances, so the assertions have teeth.
    """

    selections = (Selection.WIDEST_FIT, Selection.FIRST_FIT, Selection.FITNESS_NUMBER)
    rows = []
    failures = []

    probe = _probe_instance()
    rng = random.Random("off2b-seq-v1/probe")
    probe_seq = list(probe.items)
    rng.shuffle(probe_seq)
    probe_ids = [item.item_id for item in probe_seq]
    while probe_ids in (sorted(probe_ids), sorted(probe_ids, reverse=True)):
        rng.shuffle(probe_seq)
        probe_ids = [item.item_id for item in probe_seq]
    for ordering in Ordering:
        for selection in selections:
            for rotatable in (True, False):
                config = Config(
                    ordering=ordering, selection=selection,
                    placement=PlacementPolicy.LM, rotatable=rotatable,
                )
                placements, skyline, _m = solve_config(probe, config, sequence=list(probe_seq))
                ok = [p.item_id for p in placements] == probe_ids
                orient_ok = all(
                    sorted((p.width, p.height)) == sorted((p.original_width, p.original_height))
                    for p in placements
                )
                if rotatable:
                    orient_ok &= all(p.width == 10 for p in placements)  # w >= h normalisation or transposed fit keeps w == strip width
                else:
                    orient_ok &= all(
                        (p.width, p.height) == (p.original_width, p.original_height)
                        for p in placements
                    )
                passed = ok and orient_ok
                failures += [] if passed else [("probe", config.value)]
                rows.append(
                    {
                        "arm": "probe",
                        "instance": probe.name,
                        "config": config.value,
                        "check": "placement-order==sequence",
                        "pass": int(passed),
                    }
                )

    anchor_fail = differ = cells = 0
    for group, path in _instance_specs():
        instance = load_instance(path)
        items = list(instance.items)
        rng = random.Random(f"off2b-seq-v1/{instance.name}")
        shuffled = items[:]
        rng.shuffle(shuffled)
        for ordering in Ordering:
            config = Config(
                ordering=ordering, selection=Selection.WIDEST_FIT,
                placement=PlacementPolicy.LM,
            )
            ref_p, ref_s, _ = solve_config(instance, config)
            anchor = sorted(items, key=_ordering_key(ordering, rotatable=True))
            a_p, a_s, _ = solve_config(instance, config, sequence=anchor)
            ok = a_p == ref_p and a_s == ref_s
            if not ok:
                anchor_fail += 1
                failures.append(("anchor", instance.name, ordering.value))
            s_p, s_s, _ = solve_config(instance, config, sequence=list(shuffled))
            _validate(instance, s_p, s_s)
            changed = s_p != ref_p or s_s != ref_s
            # Contrapositive control: internal re-sorting would make every
            # shuffled-sequence run coincide with the sequence-less run.
            if changed:
                differ += 1
            cells += 1
            rows.append(
                {
                    "arm": "anchor",
                    "instance": instance.name,
                    "config": config.value,
                    "check": "sequence(sorted)==no-sequence",
                    "pass": int(ok),
                }
            )
            rows.append(
                {
                    "arm": "control",
                    "instance": instance.name,
                    "config": config.value,
                    "check": "shuffled-sequence changes outcome (evidence)",
                    "pass": int(changed),
                }
            )
        # rotatable=False anchor arm (raw-dimension keys), WIDTH only.
        config_of = Config(
            ordering=Ordering.WIDTH, selection=Selection.WIDEST_FIT,
            placement=PlacementPolicy.LM, rotatable=False,
        )
        ref_of = solve_config(instance, config_of)
        anch_of = solve_config(
            instance, config_of,
            sequence=sorted(items, key=_ordering_key(Ordering.WIDTH, rotatable=False)),
        )
        ok = anch_of[0] == ref_of[0] and anch_of[1] == ref_of[1]
        if not ok:
            anchor_fail += 1
            failures.append(("anchor+of", instance.name, Ordering.WIDTH.value))
        rows.append(
            {
                "arm": "anchor+of",
                "instance": instance.name,
                "config": config_of.value,
                "check": "sequence(sorted)==no-sequence",
                "pass": int(ok),
            }
        )
        # B-core smoke: perimeter/fitness-number/SN + tower removal on the
        # shuffled sequence must stay valid (no order assertion past TR).
        s2_p, s2_s, _ = solve_config(
            instance,
            Config(
                ordering=Ordering.PERIMETER, selection=Selection.FITNESS_NUMBER,
                placement=PlacementPolicy.SN, tower_removal=True,
            ),
            sequence=list(shuffled),
        )
        _validate(instance, s2_p, s2_s)

    with open(f"{out_dir}/accept-sequence.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["arm", "instance", "config", "check", "pass"]
        )
        writer.writeheader()
        writer.writerows(rows)
    probe_pass = sum(r["pass"] for r in rows if r["arm"] == "probe")
    probe_total = sum(1 for r in rows if r["arm"] == "probe")
    print(
        f"accept-sequence: probe {probe_pass}/{probe_total} cells exact "
        f"(placement order == given sequence, 6 orderings x 3 selections x "
        f"rotatable on/off); anchor {cells - anchor_fail}/{cells} cells exact "
        f"(+36 fixed-orientation); contrapositive control: shuffled sequence "
        f"changed the outcome in {differ}/{cells} cells"
    )
    if failures or probe_pass != probe_total or anchor_fail or differ == 0:
        print(f"FAILURES: {failures[:8]}, differ={differ}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cch.off2_validation",
        description=__doc__.splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("accept-state", "accept-sequence", "run", "summarize"):
        child = sub.add_parser(name)
        child.add_argument("--out-dir", default="docs/off2-learn")
    args = parser.parse_args(argv)
    import os

    os.makedirs(args.out_dir, exist_ok=True)
    if args.command == "accept-state":
        return _accept_state(args.out_dir)
    if args.command == "accept-sequence":
        return _accept_sequence(args.out_dir)
    if args.command == "run":
        return _run(args.out_dir)
    return _summarize(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
