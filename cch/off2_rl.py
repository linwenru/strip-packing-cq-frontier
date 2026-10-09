"""OFF-2 RL mainline v3: offline RL learning to compose components stepwise.

Design: docs/off2-learn/design-rl.md (v3; user authorized execution without
review 2026-09-25/27). Architecture returned to the v1 form that worked
(per-action models, 5 FQI rounds, narrow 3x3 actions); v2.1's shared-model +
subsample-50k regressed policy quality and stays only as a data point.

v3 (design-rl v3, user decisions 2026-09-26/27):
- data scale-up: 800 self-generated training instances (prospective
  generator, new seed range 46..145) + 160 untouched final instances
  (146..165) — RL needs far more episodes than v1's 8/instance;
- ablation arms: (i) mechanism features (buried rate / exact-fill rate /
  residual slit / neighbour imbalance); (ii) macro actions (same component
  for m=5 steps; reward still telescopes); (iii) wide action set (36);
  eps-greedy Q-hat reuse only on the main arm (recorded);
- per-fold 6h CPU budget executor wired (v2.1's missing piece).

Reward: r_t = -Δ(B+T)/(W·LB), telescoping exactly to the terminal gap
(smoke-verified on N1B). Dependency relaxation (user, 2026-09-27): stdlib +
sklearn is no longer a hard constraint for the RL line; FQI first, DQN-family
if it underperforms (torch decision deferred).

    python3 -m cch.off2_rl gen-many      # materialise training/final instances
    python3 -m cch.off2_rl cv            # folds {C, BKW, NT, PV+GEN} x variants
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from math import inf
from multiprocessing import Pool
from pathlib import Path
from statistics import mean
from time import perf_counter, process_time

from burke_bf import Instance, Item, Placement, load_instance, validate_solution

from .experiment import _instance_paths, _lower_bound
from .model import Config, Ordering, PlacementPolicy, Selection
from .off2_repair import _ns
from .off2_state import (
    _lowest_gap,
    _neighbours,
    _place_on_left,
    _raise,
    _Rectangle,
    _select,
)
from .prospective import _gen_items, _seed, GROUPS, STRIP_WIDTH
from .solver import _reduce_towers

SELECTIONS_NARROW = (Selection.WIDEST_FIT, Selection.FIRST_FIT, Selection.FITNESS_NUMBER)
SELECTIONS_WIDE = (
    Selection.WIDEST_FIT, Selection.FIRST_FIT, Selection.TRE, Selection.NRE,
    Selection.FITNESS_NUMBER, Selection.MAX_AREA,
)
POLICIES_NARROW = (PlacementPolicy.LM, PlacementPolicy.TN, PlacementPolicy.SN)
POLICIES_WIDE = (
    PlacementPolicy.LM, PlacementPolicy.RM, PlacementPolicy.TN,
    PlacementPolicy.SN, PlacementPolicy.MIN_DIFF, PlacementPolicy.MAX_DIFF,
)
FQI_ROUNDS = 5
EPISODES_PER_INSTANCE = 8  # 4 random + 2 BF-ish + 2 S2-ish
EPS_GREEDY_EPISODES = 2  # main arm only (recorded)
FQI_MAX_ROWS = 100_000
CPU_CAP_S = 21_600.0  # 6h CPU per phase (design-rl R.5)
GEN_TRAIN = range(46, 146)  # 800 self-generated training instances
GEN_FINAL = range(146, 166)  # 160 untouched final-set instances
GEN_PV_HOLD_SLICE = range(126, 146)  # held out with the PV fold (design-rl v3)

# Pool size for all collection/training pools (results are
# worker-count-independent; only wall time changes). OFF2_WORKERS=1 or 2
# keeps the machine usable (user request, 2026-10-01).
N_WORKERS = int(os.environ.get("OFF2_WORKERS", "6"))


def _action_set(wide: bool) -> tuple[list, dict]:
    sels = SELECTIONS_WIDE if wide else SELECTIONS_NARROW
    pols = POLICIES_WIDE if wide else POLICIES_NARROW
    actions = [(s, p) for s in sels for p in pols]
    configs = {
        a: Config(ordering=Ordering.WIDTH, selection=a[0], placement=a[1])
        for a in actions
    }
    return actions, configs


class _State:
    """Mutable episode state (skyline, placements, remaining, bookkeeping)."""

    __slots__ = (
        "skyline", "placements", "remaining", "h_max", "buried", "step",
        "last_dh", "exact_fills", "residual_sum",
    )

    def __init__(self, instance: Instance) -> None:
        self.skyline = [0] * instance.strip_width
        self.placements: list[Placement] = []
        self.remaining = [
            _Rectangle(it, max(it.width, it.height), min(it.width, it.height))
            for it in sorted(
                instance.items,
                key=lambda it: (-max(it.width, it.height), -min(it.width, it.height)),
            )
        ]
        self.h_max = 0
        self.buried = 0
        self.step = 0
        self.last_dh = 0
        self.exact_fills = 0
        self.residual_sum = 0


def _features(st: _State, instance: Instance, lb: int, mech: bool = False) -> list[float]:
    """15 state features; +4 mechanism features when ``mech`` (design-rl v3)."""

    sky, W = st.skyline, instance.strip_width
    segs = 1
    for a, b in zip(sky, sky[1:]):
        segs += a != b
    h_min, h_max = min(sky), max(sky)
    mean_h = sum(sky) / W
    sd_h = (sum((v - mean_h) ** 2 for v in sky) / W) ** 0.5
    gap = _lowest_gap(sky)
    rem = st.remaining
    n0 = len(instance.items)
    rem_area = sum(r.width * r.height for r in rem)
    tot_area = instance.total_area
    base = [
        segs / W,
        gap.width / W,
        h_min / lb,
        mean_h / lb,
        sd_h / lb,
        h_max / lb,
        len(rem) / n0,
        rem_area / tot_area,
        (mean(r.width for r in rem) / W) if rem else 0.0,
        (max(r.width for r in rem) / W) if rem else 0.0,
        (mean(r.height for r in rem) / lb) if rem else 0.0,
        (mean(min(r.width, r.height) * 3 <= max(r.width, r.height) for r in rem)) if rem else 0.0,
        st.buried / (W * lb),
        st.step / n0,
        st.last_dh / lb,
    ]
    if mech:
        left, right = _neighbours(sky, gap)
        lh = (left - gap.y) / lb if left != inf else 1.0
        rh = (right - gap.y) / lb if right != inf else 1.0
        base += [
            st.buried / (W * max(h_max, 1)),
            st.exact_fills / max(st.step, 1),
            (st.residual_sum / max(st.step, 1)) / W,
            abs(lh - rh),
        ]
    return base


def _step(st: _State, config: Config, sel: Selection) -> float:
    """Apply one component action; return -Δ(B+T) (caller normalises)."""

    while True:
        gap = _lowest_gap(st.skyline)
        found = _select(config, st.remaining, st.skyline, gap)
        if found is not None:
            break
        left, right = _neighbours(st.skyline, gap)
        st.buried += gap.width * (int(min(left, right)) - gap.y)
        _raise(st.skyline, gap)
    index, rect, w, h = found
    if sel is Selection.FITNESS_NUMBER:
        # ISH corner rule (mirror of off2_state._pack_loop)
        left, right = _neighbours(st.skyline, gap)
        lh = left - gap.y if left != inf else inf
        rh = right - gap.y if right != inf else inf
        fit_left = (w == gap.width) + (h == lh) + (w == gap.width and h == rh)
        fit_right = (w == gap.width) + (h == rh) + (w == gap.width and h == lh)
        on_left = fit_left > fit_right if fit_left != fit_right else lh >= rh
    else:
        on_left = _place_on_left(config.placement, st.skyline, gap, gap.y + h)
    x = gap.x if on_left else gap.right - w
    st.placements.append(
        Placement(
            item_id=rect.item.item_id, x=x, y=gap.y, width=w, height=h,
            original_width=rect.item.width, original_height=rect.item.height,
        )
    )
    st.skyline[x : x + w] = [gap.y + h] * w
    del st.remaining[index]
    st.step += 1
    st.last_dh = max(0, gap.y + h - st.h_max)
    st.h_max = max(st.h_max, gap.y + h)
    st.exact_fills += int(w == gap.width)
    st.residual_sum += gap.width - w
    W = len(st.skyline)
    return -(W * st.last_dh - w * h)


def _episode(
    instance: Instance,
    lb: int,
    policy,
    seed: int,
    actions: list,
    configs: dict,
    mech: bool = False,
    macro_m: int = 1,
) -> tuple[list[tuple], int, int]:
    """One construction episode -> (transitions, construct height, TR height).

    With macro_m > 1 the chosen action is applied to min(macro_m, remaining)
    consecutive steps; the macro reward sums the step rewards (telescoping
    unchanged). TR is applied at episode end (fixed on, not an action).
    """

    rng = random.Random(seed)
    st = _State(instance)
    W = instance.strip_width
    transitions = []
    while st.remaining:
        feats = _features(st, instance, lb, mech)
        a = policy(feats, rng)
        sel, pol = actions[a]
        config = configs[(sel, pol)]
        r_sum = 0.0
        for _ in range(min(macro_m, len(st.remaining))):
            r_sum += _step(st, config, sel)
            if not st.remaining:
                break
        r = r_sum / (W * lb)
        done = not st.remaining
        next_feats = _features(st, instance, lb, mech) if not done else None
        transitions.append((feats, a, r, next_feats, done))
    h_construct = max(st.skyline)
    placements, sky_tr, _ = _reduce_towers(
        list(st.placements), list(st.skyline), PlacementPolicy.LM
    )
    validate_solution(instance, _ns(instance, placements, max(sky_tr)))
    return transitions, h_construct, max(sky_tr)


def _behavior_policy(n_actions: int):
    return lambda _f, r: r.randrange(n_actions)


def _collect_one(task):
    """Pool worker: one behavior episode (deterministic per task)."""

    path, ep, kind, mech, macro_m, wide = task
    instance = load_instance(path)
    lb = _lower_bound(instance)
    actions, configs = _action_set(wide)
    if kind in ("bf", "s2"):
        # map the fixed rule to its action index in this action set
        target = ("widest-fit", "LM") if kind == "bf" else ("fitness-number", "SN")
        ix = [i for i, (s, p) in enumerate(actions)
              if s.value == target[0] and p.value == target[1]][0]
        policy = lambda _f, _r, _ix=ix: _ix
    else:
        policy = _behavior_policy(len(actions))
    transitions, h_c, h_tr = _episode(
        instance, lb, policy, ep, actions, configs, mech, macro_m
    )
    return instance.name, ep, kind, transitions, h_c, h_tr


def _collect(
    instances: list[tuple[str, str]],
    out_csv: str,
    mech: bool = False,
    macro_m: int = 1,
    wide: bool = False,
) -> dict[tuple[str, int], list[tuple]]:
    """Behavior episodes in a Pool (collection is not a timing measurement)."""

    tasks = [
        (path, ep, ("random", "random", "bf", "s2", "random", "random",
                    "bf", "s2")[ep], mech, macro_m, wide)
        for _group, path in instances
        for ep in range(EPISODES_PER_INSTANCE)
    ]
    episodes: dict[tuple[str, int], list[tuple]] = {}
    with Pool(N_WORKERS) as pool:
        for name, ep, kind, transitions, _hc, _ht in pool.imap_unordered(
            _collect_one, tasks, chunksize=4
        ):
            episodes[(name, ep)] = transitions
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        any_ep = next(iter(episodes.values()))
        header = (["instance", "episode", "step", "action", "reward", "done"]
                  + [f"s{i}" for i in range(len(any_ep[0][0]))])
        writer.writerow(header)
        for (name, ep), transitions in sorted(episodes.items()):
            for t, (feats, a, r, _next, done) in enumerate(transitions):
                writer.writerow([name, ep, t, a, f"{r:.6f}", int(done)]
                                + [f"{v:.6f}" for v in feats])
    print(f"wrote {out_csv}: {len(episodes)} episodes", file=sys.stderr)
    return episodes


def _one_thread_init():
    """Pool initializer: keep native libs single-threaded (N workers x 1).

    Covers OpenMP (sklearn HistGB) plus the common BLAS backends: without
    this, every per-step ``predict`` in the eps-greedy workers spawns a
    full-width thread team and the pools thrash the machine.
    """

    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _fit_one(task):
    """Pool worker: one per-action FQI fit (deterministic subsample seed)."""

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xa, y, seed, fit_seed = task
    if len(Xa) > FQI_MAX_ROWS:
        rng = np.random.default_rng(seed)
        ix = rng.choice(len(Xa), FQI_MAX_ROWS, replace=False)
        Xa, y = Xa[ix], y[ix]
    return HistGradientBoostingRegressor(
        random_state=fit_seed, early_stopping=False
    ).fit(Xa, y)


def _train_fqi(
    episodes: dict[tuple, list[tuple]], n_actions: int, mech: bool,
    seed: int = 42,
):
    """Fitted Q iteration: one HistGB model per action, 5 rounds, gamma=1.

    Per-fit transition subsample capped at FQI_MAX_ROWS; the subsample seed
    derives from (round, action, seed), so results do not depend on
    scheduling. Per-action fits are independent and run in a Pool — an
    implementation-level speedup only (design-rl v3 addendum 2). ``seed``
    is the training-randomness knob for the multi-seed robustness arm
    (OFF-3 revision step 4); the default reproduces the canonical run.
    """

    import numpy as np

    n_feat = 19 if mech else 15
    X, A, R, Xn, D = [], [], [], [], []
    for _key, ep in episodes.items():
        for feats, a, r, next_feats, done in ep:
            X.append(feats[:n_feat])
            A.append(a)
            R.append(r)
            Xn.append((next_feats if next_feats is not None else feats)[:n_feat])
            D.append(done)
    X = np.array(X, dtype=float)
    Xn = np.array(Xn, dtype=float)
    A = np.array(A, dtype=int)
    R = np.array(R, dtype=float)
    D = np.array(D, dtype=bool)

    with Pool(N_WORKERS, initializer=_one_thread_init) as pool:

        def round_fits(target, round_ix):
            tasks = [
                (X[A == a], target[A == a], seed + 1000 * round_ix + a, seed)
                for a in range(n_actions)
            ]
            return pool.map(_fit_one, tasks)

        models = round_fits(R, 0)
        for round_ix in range(1, FQI_ROUNDS):
            qn = np.zeros(len(R))
            live = ~D
            if live.any():
                qn[live] = np.max(
                    np.stack([m.predict(Xn[live]) for m in models], axis=1),
                    axis=1,
                )
            models = round_fits(R + qn, round_ix)
            print(f"  FQI round {round_ix + 1}/{FQI_ROUNDS}", file=sys.stderr,
                  flush=True)
    return models


def _greedy_policy(models):
    import numpy as np

    def policy(feats: list[float], _rng) -> int:
        x = np.array(feats, dtype=float).reshape(1, -1)
        qs = [m.predict(x)[0] for m in models]
        return int(np.argmax(qs))

    return policy


def _eps_greedy_policy(models, eps: float):
    greedy = _greedy_policy(models)

    def policy(feats: list[float], rng) -> int:
        if rng.random() < eps:
            return rng.randrange(len(models))
        return greedy(feats, rng)

    return policy


_EPS_CTX: dict = {}


def _eps_init(models, wide, mech, macro_m, seed_base=0):
    """Pool initializer: hold the frozen Q-hat models once per worker."""

    _one_thread_init()
    _EPS_CTX["models"] = models
    _EPS_CTX["actions"], _EPS_CTX["configs"] = _action_set(wide)
    _EPS_CTX["mech"] = mech
    _EPS_CTX["macro_m"] = macro_m
    _EPS_CTX["seed_base"] = seed_base


def _eps_one(task):
    """Pool worker: one eps-greedy episode (deterministic seed per task)."""

    path, ep = task
    instance = load_instance(path)
    lb = _lower_bound(instance)
    policy = _eps_greedy_policy(_EPS_CTX["models"], 0.2)
    transitions, _hc, _ht = _episode(
        instance, lb, policy, 1000 + _EPS_CTX["seed_base"] * 100000 + ep,
        _EPS_CTX["actions"], _EPS_CTX["configs"],
        mech=_EPS_CTX["mech"], macro_m=_EPS_CTX["macro_m"],
    )
    return instance.name, f"eg{ep}", transitions


def _gen_many(out_dir: str) -> list[tuple[str, str]]:
    """Materialise the self-generated instances: train 46..145, final 146..165."""

    folder = Path(out_dir) / "instances-many"
    folder.mkdir(parents=True, exist_ok=True)
    specs = []
    for g, group in enumerate(GROUPS):
        for s_idx, n in ((0, 100), (1, 300)):
            for i in GEN_TRAIN:
                seed = _seed(g, s_idx, i)
                rng = random.Random(seed)
                items = _gen_items(rng, group, n)
                path = folder / f"pv-{group}-{n}-{i:03d}.ins2D"
                if not path.exists():
                    with open(path, "w") as handle:
                        handle.write(
                            f"# off2-rl train group={group} n={n} idx={i} seed={seed}\n"
                        )
                        handle.write(f"{n}\n{STRIP_WIDTH} -1\n")
                        for j, w, h in items:
                            handle.write(f"{j} {w} {h} 1 1 0\n")
                specs.append((f"GEN-{group.upper()}{n}", str(path)))
    final_folder = Path(out_dir) / "instances-final"
    final_folder.mkdir(parents=True, exist_ok=True)
    final_specs = []
    for g, group in enumerate(GROUPS):
        for s_idx, n in ((0, 100), (1, 300)):
            for i in GEN_FINAL:
                seed = _seed(g, s_idx, i)
                rng = random.Random(seed)
                items = _gen_items(rng, group, n)
                path = final_folder / f"pv-{group}-{n}-{i:03d}.ins2D"
                if not path.exists():
                    with open(path, "w") as handle:
                        handle.write(
                            f"# off2-rl FINAL group={group} n={n} idx={i} seed={seed}\n"
                        )
                        handle.write(f"{n}\n{STRIP_WIDTH} -1\n")
                        for j, w, h in items:
                            handle.write(f"{j} {w} {h} 1 1 0\n")
                final_specs.append((f"FINAL-{group.upper()}{n}", str(path)))
    return specs, final_specs


def _load_episodes(path: str) -> dict[tuple[str, int], list[tuple]]:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return {}
    n_feat = len([c for c in rows[0]
                  if c.startswith("s") and c[1:].isdigit()])
    by_ep: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by_ep.setdefault((r["instance"], int(r["episode"])), []).append(r)
    episodes = {}
    for key, rs in by_ep.items():
        rs.sort(key=lambda r: int(r["step"]))
        ep = []
        for t, r in enumerate(rs):
            feats = [float(r[f"s{i}"]) for i in range(n_feat)]
            done = bool(int(r["done"]))
            nxt = None
            if t + 1 < len(rs):
                nxt = [float(rs[t + 1][f"s{i}"]) for i in range(n_feat)]
            ep.append((feats, int(r["action"]), float(r["reward"]), nxt, done))
        episodes[key] = ep
    return episodes


def _cv(out_dir: str) -> int:
    """Family-holdout CV with variant arms (design-rl v3).

    Folds: C / BKW / NT / PV (the PV fold holds out PV 01-15 plus the gen
    slice 126-145, recorded; gen instances are never evaluated in CV — they
    have no 864-label matrix). Variants: main (9 actions, 15 features),
    mech (+4 mechanism features), macro (m=5), wide (36 actions, folds C/NT
    only — cost control, recorded). eps-greedy Q-hat reuse on main only.
    """

    from .off2_learn import _family, _greedy_portfolio, _load_matrix

    base_specs = list(_instance_paths()) + [
        (f"PV-{g.upper()}", f"data/prospective/pv-{g}-{n}-{i:02d}.ins2D")
        for g in "abcd" for n in (100, 300) for i in range(1, 16)
    ]
    gen_specs, _final = _gen_many(out_dir)
    # Family by PATH: generated instances are named pv-* and would otherwise
    # read as the PV family (they share the generator, recorded in design-rl v3).
    family_map: dict[str, str] = {}
    gen_index: dict[str, int] = {}
    for _g, p in base_specs:
        family_map[load_instance(p).name] = _family(load_instance(p).name)
    for _g, p in gen_specs:
        name = load_instance(p).name
        family_map[name] = "GEN"
        # instance names drop the dashes (pv-a-100-046 -> PVA100046), so the
        # generator index must come from the file stem, not the name
        gen_index[name] = int(Path(p).stem.rsplit("-", 1)[1])

    variants = [
        ("main", dict(wide=False, mech=False, macro_m=1)),
        ("mech", dict(wide=False, mech=True, macro_m=1)),
        ("macro", dict(wide=False, mech=False, macro_m=5)),
        ("wide", dict(wide=True, mech=False, macro_m=1)),
    ]
    started_cpu = None
    for vname, vkw in variants:
        # the 6h CPU budget is per variant (design-rl v3): a slow variant must
        # not eat the budget of the others; truncation is reported per fold
        started_cpu = process_time()
        # collect (or reuse) behavior episodes for this variant
        ep_path = f"{out_dir}/rl-episodes-v3-{vname}.csv"
        if os.path.exists(ep_path):
            print(f"[{vname}] reuse episodes", file=sys.stderr)
            episodes = _load_episodes(ep_path)
        else:
            specs = base_specs + gen_specs
            print(f"[{vname}] collecting {len(specs)} instances x "
                  f"{EPISODES_PER_INSTANCE} episodes", file=sys.stderr)
            episodes = _collect(
                specs, ep_path, mech=vkw["mech"], macro_m=vkw["macro_m"],
                wide=vkw["wide"],
            )
        actions, configs = _action_set(vkw["wide"])
        folds = ["C", "BKW", "NT", "PV"] if vname != "wide" else ["C", "NT"]
        for held in folds:
            if process_time() - started_cpu > CPU_CAP_S:
                print(f"CPU cap reached before {vname}/{held}; truncating",
                      file=sys.stderr)
                break
            train_eps = {}
            for key, ep in episodes.items():
                fam = family_map[key[0]]
                if fam == held:
                    continue
                if held == "PV" and fam == "GEN":
                    if gen_index.get(key[0], -1) in GEN_PV_HOLD_SLICE:
                        continue
                train_eps[key] = ep
            print(f"[{vname}] fold hold-{held}: train episodes {len(train_eps)}",
                  file=sys.stderr, flush=True)
            models = _train_fqi(train_eps, len(actions), vkw["mech"])
            if vname == "main":
                train_insts = [
                    (g, p) for g, p in base_specs + gen_specs
                    if not (
                        family_map[load_instance(p).name] == held
                        or (
                            held == "PV"
                            and family_map[load_instance(p).name] == "GEN"
                            and int(Path(p).stem.split("-")[-1]) in GEN_PV_HOLD_SLICE
                        )
                    )
                ]
                eps_extra = {}
                eg_tasks = [
                    (p, ep) for _g, p in train_insts
                    for ep in range(EPS_GREEDY_EPISODES)
                ]
                with Pool(N_WORKERS, initializer=_eps_init,
                          initargs=(models, vkw["wide"], vkw["mech"],
                                    vkw["macro_m"])) as pool:
                    for name, ep_key, transitions in pool.imap_unordered(
                        _eps_one, eg_tasks, chunksize=4
                    ):
                        eps_extra[(name, ep_key)] = transitions
                # deterministic insertion order (imap_unordered arrival
                # order is scheduler-dependent; see design-rl v4 addendum)
                for key in sorted(eps_extra):
                    train_eps[key] = eps_extra[key]
                models = _train_fqi(train_eps, len(actions), vkw["mech"])
            policy = _greedy_policy(models)
            results = _eval_fold(out_dir, held, base_specs, policy, actions,
                                 configs, vkw)
            tag = f"{vname}-hold-{held}"
            print(f"  {tag}: RL={results['rl']:.3f} portfolio={results['portfolio']:.3f} "
                  f"best_fixed={results['best_fixed']:.3f} oracle={results['oracle']:.3f} "
                  f"fallback={results['fallback']:.2f}", flush=True)
            cv_path = f"{out_dir}/rl-cv-v3.csv"
            with open(cv_path, "a", newline="") as handle:
                writer = csv.writer(handle)
                if os.path.getsize(cv_path) == 0:
                    writer.writerow(["tag", "variant", "held", "rl",
                                     "portfolio", "best_fixed", "oracle",
                                     "fallback"])
                writer.writerow([tag, vname, held, f"{results['rl']:.4f}",
                                 f"{results['portfolio']:.4f}",
                                 f"{results['best_fixed']:.4f}",
                                 f"{results['oracle']:.4f}",
                                 f"{results['fallback']:.4f}"])
    print(f"wrote {out_dir}/rl-cv-v3.csv", file=sys.stderr)
    return 0


def _eval_fold(out_dir, held, base_specs, policy, actions, configs, vkw):
    """Rollout the greedy policy on the held fold's instances; baselines from
    the label matrix (fold-fair portfolio / best-fixed / oracle)."""

    from .off2_learn import _family, _greedy_portfolio, _load_matrix

    _, configs_full, matrix, _lbm = _load_matrix(out_dir)
    pv_rows = list(csv.DictReader(open(f"{out_dir}/label-matrix-pv.csv")))
    for r in pv_rows:
        matrix.setdefault(r["instance"], {})
        if isinstance(matrix[r["instance"]], list):
            continue
        matrix[r["instance"]][r["config"]] = float(r["gap_pct"])

    def yvec(name):
        v = matrix[name]
        if isinstance(v, dict):
            return [v[c] for c in configs_full]
        return list(v)

    train_names = [i for i in matrix if _family(i) != held]
    Y = {i: yvec(i) for i in train_names}
    fold_pf = _greedy_portfolio(list(Y), Y, k=12)
    best_fixed = min(range(len(configs_full)),
                     key=lambda c: mean(Y[i][c] for i in Y))
    rl_gaps, pf_gaps, bf_gaps, or_gaps, fallbacks = [], [], [], [], []
    for _g, path in base_specs:
        instance = load_instance(path)
        if _family(instance.name) != held:
            continue
        lb = _lower_bound(instance)
        _trans, _hc, h_tr = _episode(
            instance, lb, policy, 0, actions, configs,
            mech=vkw["mech"], macro_m=vkw["macro_m"],
        )
        yv = yvec(instance.name)
        rl_gap = 100.0 * (h_tr - lb) / lb
        bf = yv[best_fixed]
        fallback = rl_gap > bf
        rl_gaps.append(bf if fallback else rl_gap)
        fallbacks.append(fallback)
        pf_gaps.append(min(yv[c] for c in fold_pf))
        bf_gaps.append(bf)
        or_gaps.append(min(yv))
    return {
        "rl": mean(rl_gaps), "portfolio": mean(pf_gaps),
        "best_fixed": mean(bf_gaps), "oracle": mean(or_gaps),
        "fallback": mean(fallbacks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cch.off2_rl", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("gen-many", "cv", "final"):
        child = sub.add_parser(name)
        child.add_argument("--out-dir", default="docs/off2-learn")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.command == "gen-many":
        specs, final_specs = _gen_many(args.out_dir)
        print(f"{len(specs)} training + {len(final_specs)} final instances")
        return 0
    if args.command == "cv":
        return _cv(args.out_dir)
    print("final: pending (after CV signal)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
