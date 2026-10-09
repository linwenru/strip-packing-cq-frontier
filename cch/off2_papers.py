"""OFF-3/OFF-4 paper assets: per-instance RL gaps, combo analysis, CPU timing.

Plan: docs/off2-learn/papers-plan.md (user-approved 2026-09-29).
OFF-3 (methods paper): RL policy as the 19th/289th member of the PUBLISHED
enumeration shells (TWBF-18 / BBF-288) — never the in-review OFF-1
portfolio. OFF-4 (frontier study): unified serial CPU-time accounting.

    python3 -m cch.off2_papers rl-instances   # per-instance RL gaps + timing
    python3 -m cch.off2_papers combo          # TWBF/BBF +/- RL + bootstrap CI
    python3 -m cch.off2_papers timing         # serial CPU-time of frontier methods
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, median
from time import perf_counter, process_time

from burke_bf import load_instance

from .experiment import _instance_paths, _lower_bound
from .off2_learn import _family, _greedy_portfolio, _load_matrix
from .off2_rl import (
    EPS_GREEDY_EPISODES,
    GEN_PV_HOLD_SLICE,
    N_WORKERS,
    _action_set,
    _episode,
    _eps_init,
    _eps_one,
    _gen_many,
    _greedy_policy,
    _load_episodes,
    _train_fqi,
)

FOLDS = ("C", "BKW", "NT", "PV")
BOOTSTRAP = 10_000
BOOT_SEED = 20260917


def _base_specs() -> list[tuple[str, str]]:
    return list(_instance_paths()) + [
        (f"PV-{g.upper()}", f"data/prospective/pv-{g}-{n}-{i:02d}.ins2D")
        for g in "abcd" for n in (100, 300) for i in range(1, 16)
    ]


def _family_maps(gen_specs):
    base_specs = _base_specs()
    family_map: dict[str, str] = {}
    gen_index: dict[str, int] = {}
    for _g, p in base_specs:
        family_map[load_instance(p).name] = _family(load_instance(p).name)
    for _g, p in gen_specs:
        name = load_instance(p).name
        family_map[name] = "GEN"
        gen_index[name] = int(Path(p).stem.rsplit("-", 1)[1])
    return base_specs, family_map, gen_index


def _train_main_arm(train_eps, train_insts, actions):
    """Replicate the v3 main arm: FQI -> eps-greedy reuse -> FQI."""

    models = _train_fqi(train_eps, len(actions), False)
    eg_tasks = [
        (p, ep) for _g, p in train_insts for ep in range(EPS_GREEDY_EPISODES)
    ]
    eps_extra = {}
    with Pool(N_WORKERS, initializer=_eps_init,
              initargs=(models, False, False, 1)) as pool:
        for name, ep_key, transitions in pool.imap_unordered(
            _eps_one, eg_tasks, chunksize=4
        ):
            eps_extra[(name, ep_key)] = transitions
    # deterministic insertion order: imap_unordered arrival order is
    # scheduler-dependent and would otherwise leak into the FQI row order
    # (subsample index mapping + histogram FP accumulation)
    for key in sorted(eps_extra):
        train_eps[key] = eps_extra[key]
    return _train_fqi(train_eps, len(actions), False)


def cmd_rl_instances(out_dir: str) -> int:
    """Retrain per fold, re-roll on held instances, save per-instance rows.

    The v3 CV kept only fold means (models discarded); this regenerates the
    same deterministic policies and records per-instance gaps and rollout
    timings. Fold means are cross-checked against rl-cv-v3.csv.
    """

    gen_specs, _final = _gen_many(out_dir)
    base_specs, family_map, gen_index = _family_maps(gen_specs)
    actions, configs = _action_set(False)
    episodes = _load_episodes(f"{out_dir}/rl-episodes-v3-main.csv")
    _names, configs_full, matrix, _lbm = _load_matrix(out_dir)
    for r in csv.DictReader(open(f"{out_dir}/label-matrix-pv.csv")):
        matrix.setdefault(r["instance"], {})
        if isinstance(matrix[r["instance"]], list):
            continue
        matrix[r["instance"]][r["config"]] = float(r["gap_pct"])

    def yvec(name):
        v = matrix[name]
        return [v[c] for c in configs_full] if isinstance(v, dict) else list(v)

    out_path = f"{out_dir}/rl-instance-gaps.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "fold", "instance", "family", "lb", "h_tr", "rl_gap",
            "best_fixed", "portfolio", "oracle", "fallback", "deployed_gap",
            "rollout_cpu_s", "rollout_wall_s",
        ])
        for held in FOLDS:
            train_eps = {}
            for key, ep in episodes.items():
                fam = family_map[key[0]]
                if fam == held:
                    continue
                if held == "PV" and fam == "GEN":
                    if gen_index.get(key[0], -1) in GEN_PV_HOLD_SLICE:
                        continue
                train_eps[key] = ep
            train_insts = [
                (g, p) for g, p in base_specs + gen_specs
                if not (
                    family_map[load_instance(p).name] == held
                    or (
                        held == "PV"
                        and family_map[load_instance(p).name] == "GEN"
                        and gen_index.get(load_instance(p).name, -1)
                        in GEN_PV_HOLD_SLICE
                    )
                )
            ]
            print(f"[rl-instances] fold {held}: training", file=sys.stderr,
                  flush=True)
            models = _train_main_arm(dict(train_eps), train_insts, actions)
            with open(f"{out_dir}/rl-models-{held}.pkl", "wb") as pkl:
                pickle.dump(models, pkl)
            policy = _greedy_policy(models)
            train_names = [i for i in matrix if _family(i) != held]
            Y = {i: yvec(i) for i in train_names}
            fold_pf = _greedy_portfolio(list(Y), Y, k=12)
            best_fixed = min(range(len(configs_full)),
                             key=lambda c: mean(Y[i][c] for i in Y))
            for _g, path in base_specs:
                instance = load_instance(path)
                if _family(instance.name) != held:
                    continue
                lb = _lower_bound(instance)
                t0c, t0w = process_time(), perf_counter()
                _trans, _hc, h_tr = _episode(
                    instance, lb, policy, 0, actions, configs
                )
                cpu_s, wall_s = process_time() - t0c, perf_counter() - t0w
                yv = yvec(instance.name)
                rl_gap = 100.0 * (h_tr - lb) / lb
                bf = yv[best_fixed]
                fallback = rl_gap > bf
                writer.writerow([
                    held, instance.name, held, lb, h_tr, f"{rl_gap:.4f}",
                    f"{bf:.4f}", f"{min(yv[c] for c in fold_pf):.4f}",
                    f"{min(yv):.4f}", int(fallback),
                    f"{(bf if fallback else rl_gap):.4f}",
                    f"{cpu_s:.4f}", f"{wall_s:.4f}",
                ])
            handle.flush()
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_combo(out_dir: str) -> int:
    """OFF-3 main table: published enumeration shells +/- the RL policy.

    Primary: min(TWBF-18, RL) vs TWBF-18; secondary: min(BBF-288, RL) vs
    BBF-288. RL gap is the raw policy+TR result (no fallback — the shell
    min() subsumes it). Paired instance-level bootstrap, seed 20260917.
    """

    rl_rows = {
        r["instance"]: float(r["rl_gap"])
        for r in csv.DictReader(open(f"{out_dir}/rl-instance-gaps.csv"))
        if r["family"] != "PV"
    }
    shell: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(open("docs/stage3-baselines.csv")):
        if r["algorithm"] in ("twbf-best18", "bbf-best288"):
            shell.setdefault(r["algorithm"], {})[r["instance"]] = float(
                r["gap_pct"]
            )
    instances = sorted(rl_rows)
    out_path = f"{out_dir}/off3-combo.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "family", "rl", "twbf18", "bbf288",
                         "min_twbf_rl", "min_bbf_rl"])
        for name in instances:
            tw = shell["twbf-best18"].get(name)
            bb = shell["bbf-best288"].get(name)
            writer.writerow([
                name, _family(name), f"{rl_rows[name]:.4f}",
                f"{tw:.4f}" if tw is not None else "",
                f"{bb:.4f}" if bb is not None else "",
                f"{min(tw, rl_rows[name]):.4f}" if tw is not None else "",
                f"{min(bb, rl_rows[name]):.4f}" if bb is not None else "",
            ])
    rng = random.Random(BOOT_SEED)

    def paired_ci(diffs: list[float]) -> tuple[float, float, float]:
        boot = sorted(
            mean(rng.choice(diffs) for _ in range(len(diffs)))
            for _ in range(BOOTSTRAP)
        )
        return mean(diffs), boot[250], boot[9750]

    print(f"wrote {out_path}", file=sys.stderr)
    for algo, combo_col in (("twbf-best18", "min_twbf_rl"),
                            ("bbf-best288", "min_bbf_rl")):
        names = [n for n in instances if n in shell[algo]]
        for scope, sel in (("overall", names),
                           *[(f, [n for n in names if _family(n) == f])
                             for f in ("C", "BKW", "NT")]):
            shell_gaps = [shell[algo][n] for n in sel]
            combo_gaps = [min(shell[algo][n], rl_rows[n]) for n in sel]
            diffs = [s - c for s, c in zip(shell_gaps, combo_gaps)]
            wins = sum(d > 1e-9 for d in diffs)
            m, lo, hi = paired_ci(diffs)
            print(f"  {algo:12s} {scope:7s} n={len(sel):3d}: shell={mean(shell_gaps):.3f} "
                  f"combo={mean(combo_gaps):.3f} gain={m:+.3f} "
                  f"CI[{lo:+.3f},{hi:+.3f}] wins={wins}/{len(sel)}",
                  flush=True)
    return 0


def cmd_timing(out_dir: str) -> int:
    """OFF-4 main table: serial CPU-time of the frontier methods (3 reps,
    median; process_time primary, wall recorded). RL rollout uses the
    per-fold models saved by rl-instances."""

    from .baselines import ALGORITHMS

    wanted = {"bf-TN", "bbf-best288", "twbf-best18", "ish-core", "fh"}
    runners = {name: run for name, run in ALGORITHMS if name in wanted}
    actions, configs = _action_set(False)
    models_by_fold = {}
    for held in FOLDS:
        pkl_path = f"{out_dir}/rl-models-{held}.pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as handle:
                models_by_fold[held] = pickle.load(handle)

    out_path = f"{out_dir}/off4-timing.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["instance"], r["method"], int(r["rep"]))
                    for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "family", "method", "rep",
                             "cpu_s", "wall_s", "height"])
        for _group, path in _instance_paths():
            instance = load_instance(path)
            fam = _family(instance.name)
            lb = _lower_bound(instance)
            for method, run in sorted(runners.items()):
                for rep in range(3):
                    if (instance.name, method, rep) in done:
                        continue
                    t0c, t0w = process_time(), perf_counter()
                    placements, height = run(instance)
                    cpu_s, wall_s = process_time() - t0c, perf_counter() - t0w
                    writer.writerow([instance.name, fam, method, rep,
                                     f"{cpu_s:.4f}", f"{wall_s:.4f}", height])
                    handle.flush()
            models = models_by_fold.get(fam)
            if models is None:
                continue
            policy = _greedy_policy(models)
            for rep in range(3):
                if (instance.name, "rl-policy", rep) in done:
                    continue
                t0c, t0w = process_time(), perf_counter()
                _trans, _hc, h_tr = _episode(
                    instance, lb, policy, 0, actions, configs
                )
                cpu_s, wall_s = process_time() - t0c, perf_counter() - t0w
                writer.writerow([instance.name, fam, "rl-policy", rep,
                                 f"{cpu_s:.4f}", f"{wall_s:.4f}", h_tr])
                handle.flush()
            print(f"[timing] {instance.name} done", file=sys.stderr,
                  flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


CONTROL_SEEDS = range(5)
STAGE_BUCKETS = 10


def cmd_controls(out_dir: str) -> int:
    """OFF-3 revision step 2: learning-value control arms (external review #4).

    Arms on the 104 C/BKW/NT instances:
    - ``rand-step``: uniform random micro-action per step (5 seeds);
    - ``freq-step``: stateless sampler from the fold RL policy's marginal
      action distribution (5 seeds) — isolates state dependence;
    - ``stage-table``: fixed open-loop table (modal RL action per progress
      bucket, learned on the fold's training instances) — open vs closed
      loop;
    - ``instance-level``: per-instance one-shot best fixed config (label
      matrix) — stepwise vs one-shot.
    Each arm is composed into the published shells (min per instance) with
    paired bootstrap CIs, exactly like the RL combo.
    """

    actions, configs = _action_set(False)
    base104 = list(_instance_paths())
    gen_specs, _final = _gen_many(out_dir)
    base_specs, family_map, gen_index = _family_maps(gen_specs)
    models_by_fold = {}
    for held in ("C", "BKW", "NT"):
        with open(f"{out_dir}/rl-models-{held}.pkl", "rb") as handle:
            models_by_fold[held] = pickle.load(handle)

    # ---- phase 1: logged rollouts of the canonical policies on the 104
    # (action histograms per fold; sanity-check raw gaps vs rl-instance-gaps)
    hist_by_fold: dict[str, list[int]] = {}
    for _g, path in base104:
        instance = load_instance(path)
        fam = _family(instance.name)
        lb = _lower_bound(instance)
        policy = _greedy_policy(models_by_fold[fam])
        transitions, _hc, h_tr = _episode(instance, lb, policy, 0, actions,
                                          configs)
        for _feats, a, _r, _nxt, _done in transitions:
            hist_by_fold.setdefault(fam, []).append(a)
    print("[controls] phase 1 done (eval rollouts + histograms)",
          file=sys.stderr, flush=True)

    # ---- phase 2: logged rollouts on each fold's training instances
    # (stage-table action counts per progress bucket)
    stage_counts: dict[str, list[list[int]]] = {}
    for held in ("C", "BKW", "NT"):
        counts = [[0] * len(actions) for _ in range(STAGE_BUCKETS)]
        train_paths = [
            p for _g, p in base_specs + gen_specs
            if family_map[load_instance(p).name] != held
        ]
        for path in train_paths:
            instance = load_instance(path)
            lb = _lower_bound(instance)
            policy = _greedy_policy(models_by_fold[held])
            transitions, _hc, _ht = _episode(instance, lb, policy, 0, actions,
                                             configs)
            for feats, a, _r, _nxt, _done in transitions:
                bucket = min(STAGE_BUCKETS - 1,
                             int(feats[13] * STAGE_BUCKETS))
                counts[bucket][a] += 1
        stage_counts[held] = counts
        print(f"[controls] phase 2 fold {held}: {len(train_paths)} training "
              f"rollouts", file=sys.stderr, flush=True)
    stage_table = {
        held: [max(range(len(actions)), key=lambda a, _c=counts: _c[a])
               for counts in stage_counts[held]]
        for held in stage_counts
    }

    # ---- phase 3: control-arm rollouts on the 104
    out_path = f"{out_dir}/off3-controls.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["arm"], r["instance"], int(r["seed"]))
                    for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["arm", "instance", "family", "seed", "gap"])
        for _g, path in base104:
            instance = load_instance(path)
            fam = _family(instance.name)
            lb = _lower_bound(instance)
            n_actions = len(actions)
            total = hist_by_fold[fam]
            cdf = []
            acc = 0
            for a in range(n_actions):
                acc += total.count(a)
                cdf.append(acc / len(total))
            table = stage_table[fam]
            for seed in CONTROL_SEEDS:
                for arm in ("rand-step", "freq-step"):
                    if (arm, instance.name, seed) in done:
                        continue
                    if arm == "rand-step":
                        policy = lambda _f, r: r.randrange(n_actions)
                    else:
                        def policy(_f, r, _cdf=cdf):
                            x = r.random()
                            for a, c in enumerate(_cdf):
                                if x <= c:
                                    return a
                            return len(_cdf) - 1
                    transitions, _hc, h_tr = _episode(
                        instance, lb, policy, seed, actions, configs
                    )
                    writer.writerow([arm, instance.name, fam, seed,
                                     f"{100.0 * (h_tr - lb) / lb:.4f}"])
                    handle.flush()
            if ("stage-table", instance.name, 0) not in done:
                def policy(feats, _r, _table=table):
                    return _table[min(STAGE_BUCKETS - 1,
                                      int(feats[13] * STAGE_BUCKETS))]
                transitions, _hc, h_tr = _episode(
                    instance, lb, policy, 0, actions, configs
                )
                writer.writerow(["stage-table", instance.name, fam, 0,
                                 f"{100.0 * (h_tr - lb) / lb:.4f}"])
                handle.flush()
        print("[controls] phase 3 done", file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


MULTI_SEEDS = (43, 44, 45)


def cmd_multiseed(out_dir: str) -> int:
    """OFF-3 revision step 4a: training-randomness robustness.

    For each extra seed x fold, rerun the full main-arm pipeline (FQI +
    seed-shifted eps-greedy reuse + refit) and re-roll the held fold.
    The behaviour-episode pool is shared (fixed dataset); the seeds vary
    only the fit-level and reuse-collection randomness.
    """

    gen_specs, _final = _gen_many(out_dir)
    base_specs, family_map, gen_index = _family_maps(gen_specs)
    actions, configs = _action_set(False)
    episodes = _load_episodes(f"{out_dir}/rl-episodes-v3-main.csv")
    _names, configs_full, matrix, _lbm = _load_matrix(out_dir)
    for r in csv.DictReader(open(f"{out_dir}/label-matrix-pv.csv")):
        matrix.setdefault(r["instance"], {})
        if isinstance(matrix[r["instance"]], list):
            continue
        matrix[r["instance"]][r["config"]] = float(r["gap_pct"])

    def yvec(name):
        v = matrix[name]
        return [v[c] for c in configs_full] if isinstance(v, dict) else list(v)

    out_path = f"{out_dir}/off3-multiseed.csv"
    done_folds = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done_folds = {(int(r["seed"]), r["fold"])
                          for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["seed", "fold", "instance", "rl_gap",
                             "best_fixed", "fallback", "deployed_gap"])
        for seed in MULTI_SEEDS:
            for held in FOLDS:
                if (seed, held) in done_folds:
                    continue
                train_eps = {}
                for key, ep in episodes.items():
                    fam = family_map[key[0]]
                    if fam == held:
                        continue
                    if held == "PV" and fam == "GEN":
                        if gen_index.get(key[0], -1) in GEN_PV_HOLD_SLICE:
                            continue
                    train_eps[key] = ep
                models = _train_fqi(train_eps, len(actions), False,
                                    seed=seed)
                train_insts = [
                    (g, p) for g, p in base_specs + gen_specs
                    if not (
                        family_map[load_instance(p).name] == held
                        or (
                            held == "PV"
                            and family_map[load_instance(p).name] == "GEN"
                            and gen_index.get(load_instance(p).name, -1)
                            in GEN_PV_HOLD_SLICE
                        )
                    )
                ]
                eg_tasks = [
                    (p, ep) for _g, p in train_insts
                    for ep in range(EPS_GREEDY_EPISODES)
                ]
                eps_extra = {}
                with Pool(N_WORKERS, initializer=_eps_init,
                          initargs=(models, False, False, 1, seed)) as pool:
                    for name, ep_key, transitions in pool.imap_unordered(
                        _eps_one, eg_tasks, chunksize=4
                    ):
                        eps_extra[(name, ep_key)] = transitions
                for key in sorted(eps_extra):
                    train_eps[key] = eps_extra[key]
                models = _train_fqi(train_eps, len(actions), False,
                                    seed=seed)
                policy = _greedy_policy(models)
                train_names = [i for i in matrix if _family(i) != held]
                best_fixed = min(
                    range(len(configs_full)),
                    key=lambda c: mean(yvec(i)[c] for i in train_names),
                )
                for _g, path in base_specs:
                    instance = load_instance(path)
                    if _family(instance.name) != held:
                        continue
                    lb = _lower_bound(instance)
                    _trans, _hc, h_tr = _episode(
                        instance, lb, policy, 0, actions, configs
                    )
                    rl_gap = 100.0 * (h_tr - lb) / lb
                    bf = yvec(instance.name)[best_fixed]
                    fallback = rl_gap > bf
                    writer.writerow([seed, held, instance.name,
                                     f"{rl_gap:.4f}", f"{bf:.4f}",
                                     int(fallback),
                                     f"{(bf if fallback else rl_gap):.4f}"])
                handle.flush()
                print(f"[multiseed] seed={seed} fold={held} done",
                      file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_final_eval(out_dir: str) -> int:
    """OFF-3 revision step 4b: the untouched 160-instance final set.

    Train the main arm on ALL development instances (no holdout — the
    final set is the holdout), then evaluate once on the 160 final
    instances: RL raw + published shells (bf-TN / twbf-18 / bbf-288) +
    LS@5s reference + shell combos. No fallback (no label matrix on the
    final set; reported raw, recorded).
    """

    from .baselines import ALGORITHMS
    from .model import Config, Ordering, PlacementPolicy, Selection
    from .off2_state import solve_config_traced
    from .shell import _sort_key

    episodes = _load_episodes(f"{out_dir}/rl-episodes-v3-main.csv")
    actions, configs = _action_set(False)
    print(f"[final-eval] training on all {len(episodes)} episodes",
          file=sys.stderr, flush=True)
    models = _train_fqi(episodes, len(actions), False)
    gen_specs, final_specs = _gen_many(out_dir)
    base_specs, family_map, gen_index = _family_maps(gen_specs)
    eg_tasks = [
        (p, ep) for _g, p in base_specs + gen_specs
        for ep in range(EPS_GREEDY_EPISODES)
    ]
    eps_extra = {}
    with Pool(N_WORKERS, initializer=_eps_init,
              initargs=(models, False, False, 1, 0)) as pool:
        for name, ep_key, transitions in pool.imap_unordered(
            _eps_one, eg_tasks, chunksize=4
        ):
            eps_extra[(name, ep_key)] = transitions
    for key in sorted(eps_extra):
        episodes[key] = eps_extra[key]
    models = _train_fqi(episodes, len(actions), False)
    policy = _greedy_policy(models)
    print("[final-eval] training done; evaluating 160 finals",
          file=sys.stderr, flush=True)

    runners = {n: r for n, r in ALGORITHMS
               if n in ("bf-TN", "twbf-best18", "bbf-best288")}
    ls_config = Config(
        ordering=Ordering.PERIMETER, selection=Selection.FITNESS_NUMBER,
        placement=PlacementPolicy.SN, tower_removal=True,
    )

    def ls5(instance, lb):
        items = list(instance.items)
        rng = random.Random(3000)
        current = sorted(items,
                         key=lambda it: _sort_key(Ordering.PERIMETER, it))
        t0 = process_time()
        _p, sky, _m, _t = solve_config_traced(instance, ls_config,
                                              sequence=current)
        cur_h = best_h = max(sky)
        while process_time() - t0 < 5.0 and best_h > lb:
            a, b = rng.sample(range(len(items)), 2)
            cand = list(current)
            cand[a], cand[b] = cand[b], cand[a]
            _p, sky, _m, _t = solve_config_traced(instance, ls_config,
                                                  sequence=cand)
            h = max(sky)
            if h <= cur_h:
                current, cur_h = cand, h
                if h < best_h:
                    best_h = h
        return best_h

    out_path = f"{out_dir}/off3-final.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {r["instance"] for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "group", "n", "lb", "rl_raw",
                             "bf_tn", "twbf18", "bbf288", "ls5",
                             "min_twbf_rl", "min_bbf_rl"])
        for _g, path in final_specs:
            instance = load_instance(path)
            if instance.name in done:
                continue
            lb = _lower_bound(instance)
            _trans, _hc, h_tr = _episode(instance, lb, policy, 0, actions,
                                         configs)
            rl_gap = 100.0 * (h_tr - lb) / lb
            heights = {}
            for mname, run in runners.items():
                _pl, h = run(instance)
                heights[mname] = 100.0 * (h - lb) / lb
            h_ls = 100.0 * (ls5(instance, lb) - lb) / lb
            writer.writerow([
                instance.name, instance.name[:3], len(instance.items), lb,
                f"{rl_gap:.4f}", f"{heights['bf-TN']:.4f}",
                f"{heights['twbf-best18']:.4f}", f"{heights['bbf-best288']:.4f}",
                f"{h_ls:.4f}",
                f"{min(heights['twbf-best18'], rl_gap):.4f}",
                f"{min(heights['bbf-best288'], rl_gap):.4f}",
            ])
            handle.flush()
            print(f"[final-eval] {instance.name} done", file=sys.stderr,
                  flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_figures(out_dir: str) -> int:
    """OFF-3 figures: quality-time frontier, decision-case layouts.

    fig-frontier.pdf: cost-quality plane (log x) with the frontier
    staircase, dominated points marked. fig-case.pdf: side-by-side layouts
    of the learned policy vs the TWBF-18 shell's best member on N1A (an
    intersection win). matplotlib is an OFF-2 exploratory dependency.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig_dir = Path(out_dir).parent / "off3-learned-composer" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- frontier figure
    lb_map = {}
    for r in csv.DictReader(open(f"{out_dir}/label-matrix.csv")):
        lb_map[r["instance"]] = int(r["lb"])
    tim = list(csv.DictReader(open(f"{out_dir}/off4-timing.csv")))
    # method -> (sum of per-instance median cpu, weighted mean gap)
    by_inst: dict[str, dict[str, list]] = {}
    for r in tim:
        by_inst.setdefault(r["instance"], {}).setdefault(
            r["method"], []).append((float(r["cpu_s"]),
                                     100.0 * (int(r["height"]) - lb_map[r["instance"]])
                                     / lb_map[r["instance"]]))

    def agg(method):
        cpus, gaps = [], []
        for inst, per in by_inst.items():
            if method not in per:
                continue
            rs = sorted(per[method])
            cpus.append(rs[1][0])
            gaps.append(rs[1][1])
        return sum(cpus) / len(cpus), mean(gaps)

    base_pts = {m: agg(m) for m in ("bf-TN", "ish-core", "twbf-best18",
                                    "bbf-best288", "fh")}
    rl1 = list(csv.DictReader(open(f"{out_dir}/off4-timing-rl-1t.csv")))
    rl_cpu = mean(sorted(
        [(float(r["cpu_s"])) for r in rl1 if r["instance"] == i]
    )[1] for i in {r["instance"] for r in rl1})
    rl_gap = mean(
        100.0 * (int(r["height"]) - lb_map[r["instance"]]) / lb_map[r["instance"]]
        for r in rl1 if r["rep"] == "0"
    )
    base_pts["rl-policy (clean)"] = (rl_cpu, rl_gap)
    combo_gap = mean(
        float(r["min_bbf_rl"]) for r in csv.DictReader(
            open(f"{out_dir}/off3-combo.csv")) if r["bbf288"]
    )
    bkw13_combo = None  # combo CSV is n=103 for min_bbf_rl; add BKW13
    tim13 = [r for r in tim
             if r["method"] == "bbf-best288" and r["instance"] == "BKW13"]
    if tim13:
        g13 = 100.0 * (int(tim13[0]["height"]) - lb_map["BKW13"]) / lb_map["BKW13"]
        rl13 = [r for r in rl1 if r["instance"] == "BKW13"]
        g13rl = 100.0 * (int(rl13[0]["height"]) - lb_map["BKW13"]) / lb_map["BKW13"]
        combo_gap = (combo_gap * 103 + min(g13, g13rl)) / 104
    base_pts["bbf-288+RL"] = (
        agg("bbf-best288")[0] + rl_cpu, combo_gap)
    pf_rows = list(csv.DictReader(open(f"{out_dir}/off3-portfolio.csv")))
    base_pts["greedy-12 portfolio"] = (
        mean(float(r["cpu_s"]) for r in pf_rows),
        mean(float(r["portfolio_gap"]) for r in pf_rows),
    )
    curve_rows = list(csv.DictReader(open(f"{out_dir}/off3-ls-curve.csv")))
    curve = {}
    for r in curve_rows:
        curve.setdefault(float(r["budget_s"]), []).append(
            (float(r["cpu_s"]), float(r["gap"])))
    ls_pts = [(mean(c for c, _g in v), mean(g for _c, g in v))
              for _b, v in sorted(curve.items())]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xs = [p[0] for p in ls_pts]
    ys = [p[1] for p in ls_pts]
    ax.plot(xs, ys, "o-", color="tab:blue", lw=1.5, ms=5,
            label="budgeted local search")
    for m, (c, g) in base_pts.items():
        ax.plot(c, g, "s", ms=7, label=m)
        ax.annotate(m, (c, g), textcoords="offset points", xytext=(6, 4),
                    fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("CPU time per instance (s, log scale)")
    ax.set_ylabel("mean gap (%)")
    ax.set_title("Cost–quality plane, 104 instances (family-weighted)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig-frontier.pdf")
    plt.close(fig)

    # ---- decision-case figure: N1A, RL vs TWBF-18 best member
    import zlib

    import twbf as twbf_pkg

    from .model import PlacementPolicy

    instance = load_instance("data/nt/N1a.ins2D")
    lb = _lower_bound(instance)
    with open(f"{out_dir}/rl-models-NT.pkl", "rb") as fh:
        models = pickle.load(fh)
    policy = _greedy_policy(models)
    actions, configs = _action_set(False)
    from . import off2_rl as rl

    st = rl._State(instance)
    rng = random.Random(0)
    while st.remaining:
        feats = rl._features(st, instance, lb)
        a = policy(feats, rng)
        sel, pol = actions[a]
        rl._step(st, configs[(sel, pol)], sel)
    rl_pl, rl_sky, _m = rl._reduce_towers(
        list(st.placements), list(st.skyline), PlacementPolicy.LM
    )
    rl_h = max(rl_sky)
    best_sol, _h = twbf_pkg.solve(instance)
    tw_pl, tw_h = list(best_sol.placements), best_sol.height

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    for ax, placements, height, title in (
        (axes[0], tw_pl, tw_h, f"TWBF-18 best member (H={tw_h})"),
        (axes[1], rl_pl, rl_h, f"learned policy (H={rl_h})"),
    ):
        for p in placements:
            ax.add_patch(Rectangle((p.x, p.y), p.width, p.height,
                                   facecolor=f"#{zlib.crc32(p.item_id.encode()) & 0xFFFFFF:06x}",
                                   edgecolor="black", lw=0.3, alpha=0.85))
        ax.set_xlim(0, instance.strip_width)
        ax.set_ylim(0, max(tw_h, rl_h) * 1.05)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")  # identical x/y scale: rotated items keep size
    fig.tight_layout()
    fig.savefig(fig_dir / "fig-case.pdf")
    plt.close(fig)
    print(f"wrote {fig_dir}/fig-frontier.pdf, fig-case.pdf",
          file=sys.stderr)
    return 0


def cmd_round2(out_dir: str) -> int:
    """OFF-3 review round 2: three small experiments.

    (i) fallback on the final set: the fallback config is the all-dev
    best-fixed (train-side choice; no final labels needed) — run it on the
    160 finals and report deployed = min(RL raw, fixed).
    (ii) key controls on the final set (exploratory, disclosed): rand-step /
    freq-step / stage-table vs the RL policy, composed into the shells.
    (iii) TR-gain policy dependence: TR delta of RL rollouts vs the
    stage2-best fixed config (stage1 vs stage2-tower CSVs).
    """

    from .model import Config, Ordering, PlacementPolicy, Selection
    from .solver import solve_config

    # ---- (iii) TR-gain policy dependence (cheap; do first)
    lb_map, stage1, stage2t = {}, {}, {}
    with open("docs/stage1-runs.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            key = f"{r['ordering']}/{r['selection']}/{r['placement']}"
            stage1[(r["instance"], key)] = (float(r["gap_pct"]),
                                            int(r["lower_bound"]))
            lb_map[r["instance"]] = int(r["lower_bound"])
    with open("docs/stage2-tower-runs.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            key = f"{r['ordering']}/{r['selection']}/{r['placement']}"
            stage2t[(r["instance"], key)] = float(r["gap_pct"])
    cfg_key = "perimeter/fitness-number/SN"
    deltas_fixed = [
        stage2t[(i, cfg_key)] - stage1[(i, cfg_key)][0]
        for i in lb_map if (i, cfg_key) in stage2t
    ]
    print(f"[round2] TR delta, fixed {cfg_key}: mean {mean(deltas_fixed):+.3f}pp "
          f"(n={len(deltas_fixed)})", file=sys.stderr, flush=True)

    # RL TR delta: re-roll with canonical fold models, recording construct vs TR
    actions, configs = _action_set(False)
    rl_deltas = []
    for _g, path in _instance_paths():
        instance = load_instance(path)
        fam = _family(instance.name)
        with open(f"{out_dir}/rl-models-{fam}.pkl", "rb") as fh:
            models = pickle.load(fh)
        lb = _lower_bound(instance)
        policy = _greedy_policy(models)
        _t, h_c, h_tr = _episode(instance, lb, policy, 0, actions, configs)
        rl_deltas.append(100.0 * (h_tr - h_c) / lb)
    print(f"[round2] TR delta, RL policy: mean {mean(rl_deltas):+.3f}pp "
          f"(n={len(rl_deltas)})", file=sys.stderr, flush=True)

    # ---- (i)+(ii): retrain the all-dev final model (canonical seed) and save
    episodes = _load_episodes(f"{out_dir}/rl-episodes-v3-main.csv")
    models = _train_fqi(episodes, len(actions), False)
    gen_specs, final_specs = _gen_many(out_dir)
    base_specs, family_map, gen_index = _family_maps(gen_specs)
    eg_tasks = [
        (p, ep) for _g, p in base_specs + gen_specs
        for ep in range(EPS_GREEDY_EPISODES)
    ]
    eps_extra = {}
    with Pool(N_WORKERS, initializer=_eps_init,
              initargs=(models, False, False, 1, 0)) as pool:
        for name, ep_key, transitions in pool.imap_unordered(
            _eps_one, eg_tasks, chunksize=4
        ):
            eps_extra[(name, ep_key)] = transitions
    for key in sorted(eps_extra):
        episodes[key] = eps_extra[key]
    models = _train_fqi(episodes, len(actions), False)
    with open(f"{out_dir}/rl-models-final.pkl", "wb") as fh:
        pickle.dump(models, fh)
    policy = _greedy_policy(models)
    print("[round2] final model retrained+saved", file=sys.stderr,
          flush=True)

    # (i) fallback on finals: all-dev best-fixed config
    _names, configs_full, matrix, _lbm = _load_matrix(out_dir)
    for r in csv.DictReader(open(f"{out_dir}/label-matrix-pv.csv")):
        matrix.setdefault(r["instance"], {})
        if isinstance(matrix[r["instance"]], list):
            continue
        matrix[r["instance"]][r["config"]] = float(r["gap_pct"])

    def yvec(name):
        v = matrix[name]
        return [v[c] for c in configs_full] if isinstance(v, dict) else list(v)

    best_c = min(range(len(configs_full)),
                 key=lambda c: mean(yvec(i)[c] for i in matrix))
    cfg_str = configs_full[best_c]
    o, s, rest = cfg_str.split("/")
    p, tw = rest.split("+tw")[0], rest.split("+tw")[1][0]
    vn = rest.split("+vn")[1][0]
    fallback_cfg = Config(
        ordering=Ordering(o), selection=Selection(s),
        placement=PlacementPolicy(p), tower_removal=(tw == "T"),
        vertical_niche=(vn == "T"),
    )
    print(f"[round2] all-dev best-fixed config: {cfg_str}", file=sys.stderr,
          flush=True)

    # (ii) controls: freq histogram + stage table from the final model's
    # rollouts on the dev base instances (exploratory controls, disclosed)
    hist: list[int] = []
    stage_counts = [[0] * len(actions) for _ in range(STAGE_BUCKETS)]
    for _g, path in base_specs:
        instance = load_instance(path)
        lb = _lower_bound(instance)
        transitions, _hc, _ht = _episode(instance, lb, policy, 0, actions,
                                         configs)
        for feats, a, _r, _nxt, _done in transitions:
            hist.append(a)
            bucket = min(STAGE_BUCKETS - 1, int(feats[13] * STAGE_BUCKETS))
            stage_counts[bucket][a] += 1
    table = [max(range(len(actions)), key=lambda a, _c=c: _c[a])
             for c in stage_counts]
    cdf = []
    acc = 0
    for a in range(len(actions)):
        acc += hist.count(a)
        cdf.append(acc / len(hist))

    out_path = f"{out_dir}/off3-round2.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["arm"], r["instance"], int(r["seed"]))
                    for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["arm", "instance", "seed", "gap"])
        for _g, path in final_specs:
            instance = load_instance(path)
            lb = _lower_bound(instance)
            # fallback member
            if ("fixed", instance.name, 0) not in done:
                placements, skyline, _ = solve_config(instance, fallback_cfg)
                writer.writerow(["fixed", instance.name, 0,
                                 f"{100.0 * (max(skyline) - lb) / lb:.4f}"])
                handle.flush()
            for seed in CONTROL_SEEDS:
                for arm in ("rand-step", "freq-step"):
                    if (arm, instance.name, seed) in done:
                        continue
                    if arm == "rand-step":
                        pol = lambda _f, r: r.randrange(len(actions))
                    else:
                        def pol(_f, r, _cdf=cdf):
                            x = r.random()
                            for a, c in enumerate(_cdf):
                                if x <= c:
                                    return a
                            return len(_cdf) - 1
                    _t, _hc, h_tr = _episode(instance, lb, pol, seed,
                                             actions, configs)
                    writer.writerow([arm, instance.name, seed,
                                     f"{100.0 * (h_tr - lb) / lb:.4f}"])
                    handle.flush()
            if ("stage-table", instance.name, 0) not in done:
                def pol(feats, _r, _table=table):
                    return _table[min(STAGE_BUCKETS - 1,
                                      int(feats[13] * STAGE_BUCKETS))]
                _t, _hc, h_tr = _episode(instance, lb, pol, 0, actions,
                                         configs)
                writer.writerow(["stage-table", instance.name, 0,
                                 f"{100.0 * (h_tr - lb) / lb:.4f}"])
                handle.flush()
        print("[round2] final-set controls done", file=sys.stderr,
              flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_round3(out_dir: str) -> int:
    """OFF-3 review round 3: fold-fair greedy-12 portfolio timing+gaps.

    The portfolio is a train-selected deterministic constructive ensemble
    and belongs in the main comparison. Per fold (C/BKW/NT): recompute the
    greedy-12 on training instances, run each member on the held instances
    (3 serial reps, process_time), verify gaps equal the label matrix, and
    record per-instance portfolio gap and cost.
    """

    from .model import Config, Ordering, PlacementPolicy, Selection
    from .solver import solve_config

    _names, configs_full, matrix, _lbm = _load_matrix(out_dir)
    # match the CV convention: PV labels join the training pool when PV is
    # not the held family (as _eval_fold does)
    for r in csv.DictReader(open(f"{out_dir}/label-matrix-pv.csv")):
        matrix.setdefault(r["instance"], {})
        if isinstance(matrix[r["instance"]], list):
            continue
        matrix[r["instance"]][r["config"]] = float(r["gap_pct"])

    def yvec(name):
        v = matrix[name]
        return [v[c] for c in configs_full] if isinstance(v, dict) else list(v)

    out_path = f"{out_dir}/off3-portfolio.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {r["instance"] for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["fold", "instance", "portfolio_gap", "cpu_s",
                             "wall_s", "verified"])
        for held in ("C", "BKW", "NT"):
            train_names = [i for i in matrix if _family(i) != held]
            Y = {i: yvec(i) for i in train_names}
            pf = _greedy_portfolio(list(Y), Y, k=12)
            for _g, path in _instance_paths():
                instance = load_instance(path)
                if _family(instance.name) != held or instance.name in done:
                    continue
                lb = _lower_bound(instance)
                best_gap, cpu_med = None, None
                times = []
                for rep in range(3):
                    best_h = None
                    t0c, t0w = process_time(), perf_counter()
                    for c in pf:
                        o, s, rest = configs_full[c].split("/")
                        p, tw = rest.split("+tw")[0], rest.split("+tw")[1][0]
                        vn = rest.split("+vn")[1][0]
                        cfg = Config(
                            ordering=Ordering(o), selection=Selection(s),
                            placement=PlacementPolicy(p),
                            tower_removal=(tw == "T"),
                            vertical_niche=(vn == "T"),
                        )
                        placements, skyline, _ = solve_config(instance, cfg)
                        h = max(skyline)
                        if best_h is None or h < best_h:
                            best_h = h
                    times.append((process_time() - t0c, perf_counter() - t0w))
                best_gap = 100.0 * (best_h - lb) / lb
                times.sort()
                cpu_med, wall_med = times[1]
                yv = yvec(instance.name)
                # the label matrix stores 4-decimal rounded gaps
                verified = abs(best_gap - min(yv[c] for c in pf)) < 5e-4
                writer.writerow([held, instance.name, f"{best_gap:.4f}",
                                 f"{cpu_med:.4f}", f"{wall_med:.4f}",
                                 int(verified)])
                handle.flush()
            print(f"[portfolio] fold {held} done", file=sys.stderr,
                  flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_ish_curve(out_dir: str) -> int:
    """OFF-4: the PUBLISHED search-kernel curve (ISH RandomLS, Wei et al. 2017
    Algorithm 2) at wall budgets 0.2/1/5/20/60s on the 104 instances.

    Replaces the in-review stage2-best-core curve with a fully published
    kernel. RandomLS uses wall-clock limits internally (faithful); CPU time
    is recorded alongside (disclosed in the paper).
    """

    from ish.rls import random_ls

    out_path = f"{out_dir}/off4-ish-curve.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["instance"], float(r["budget_s"]))
                    for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "family", "budget_s", "gap",
                             "wall_s", "cpu_s"])
        for _g, path in _instance_paths():
            instance = load_instance(path)
            fam = _family(instance.name)
            lb = _lower_bound(instance)
            for budget in LS_CURVE_BUDGETS:
                if (instance.name, budget) in done:
                    continue
                t0c = process_time()
                best_h, _pl, elapsed = random_ls(
                    instance, time_limit=budget, seed=3000
                )
                cpu_s = process_time() - t0c
                writer.writerow([instance.name, fam, f"{budget:.1f}",
                                 f"{100.0 * (best_h - lb) / lb:.4f}",
                                 f"{elapsed:.3f}", f"{cpu_s:.3f}"])
                handle.flush()
            print(f"[ish-curve] {instance.name} done", file=sys.stderr,
                  flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_figures_off4(out_dir: str) -> int:
    """OFF-4 frontier figure: cost-quality plane with the published
    RandomLS curve as the headline and the concurrent-work-core curve dashed
    for reference."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir).parent / "off4-cq-frontier" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lb_map = {}
    for r in csv.DictReader(open(f"{out_dir}/label-matrix.csv")):
        lb_map[r["instance"]] = int(r["lb"])
    tim = list(csv.DictReader(open(f"{out_dir}/off4-timing.csv")))
    by_inst: dict[str, dict[str, list]] = {}
    for r in tim:
        by_inst.setdefault(r["instance"], {}).setdefault(
            r["method"], []).append((float(r["cpu_s"]),
                                     100.0 * (int(r["height"]) - lb_map[r["instance"]])
                                     / lb_map[r["instance"]]))

    def agg(method):
        cpus, gaps = [], []
        for inst, per in by_inst.items():
            if method not in per:
                continue
            rs = sorted(per[method])
            cpus.append(rs[1][0])
            gaps.append(rs[1][1])
        return sum(cpus) / len(cpus), mean(gaps)

    base_pts = {m: agg(m) for m in ("bf-TN", "ish-core", "fh")}
    rl1 = list(csv.DictReader(open(f"{out_dir}/off4-timing-rl-1t.csv")))
    rl_cpu = mean(sorted(
        [float(r["cpu_s"]) for r in rl1 if r["instance"] == i]
    )[1] for i in {r["instance"] for r in rl1})
    rl_gap = mean(
        100.0 * (int(r["height"]) - lb_map[r["instance"]]) / lb_map[r["instance"]]
        for r in rl1 if r["rep"] == "0"
    )
    base_pts["rl-policy (capped)"] = (rl_cpu, rl_gap)
    # same-source rule: shell full points from the incremental runs'
    # endpoints (median of 3 continuous runs), shared with the prefix curves
    inc_end: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"{out_dir}/off4-enum-incremental3.csv")):
        inc_end[r["shell"]][r["instance"]].append(
            (int(r["member"]), float(r["cum_cpu_s"]), float(r["best_gap"])))
    for sh, label in (("twbf-18", "twbf-18"), ("bbf-288", "bbf-288"),
                      ("greedy-12", "greedy-12")):
        cpus, gaps = [], []
        for inst, rows in inc_end[sh].items():
            last = max(m for m, _c, _g in rows)
            end = [ (c, g) for m, c, g in rows if m == last ]
            cpus.append(mean(c for c, _g in end))
            gaps.append(end[0][1])
        base_pts[label] = (mean(cpus), mean(gaps))
    bbf_end_mean = base_pts["bbf-288"][0]
    base_pts["bbf-288+RL"] = (bbf_end_mean + rl_cpu, 4.405)

    ish_rows = list(csv.DictReader(open(f"{out_dir}/off4-ish-curve.csv")))
    ish = {}
    for r in ish_rows:
        ish.setdefault(float(r["budget_s"]), []).append(
            (float(r["wall_s"]), float(r["gap"])))
    dense = defaultdict(list)
    for r in csv.DictReader(open(f"{out_dir}/off4-ish-dense.csv")):
        dense[float(r["budget_s"])].append(
            (float(r["cpu_s"]), float(r["gap"])))
    dense_pts = []
    dense_lo, dense_hi = [], []
    by_b_seed = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"{out_dir}/off4-ish-dense.csv")):
        by_b_seed[float(r["budget_s"])][int(r["seed"])].append(
            (float(r["cpu_s"]), float(r["gap"])))
    for b in sorted(by_b_seed):
        seed_means = [mean(g for _c, g in v) for v in by_b_seed[b].values()]
        cpu_mean = mean(c for v in by_b_seed[b].values() for c, _g in v)
        dense_pts.append((cpu_mean, mean(seed_means)))
        dense_lo.append((cpu_mean, min(seed_means)))
        dense_hi.append((cpu_mean, max(seed_means)))
    ish_pts = [(mean(c for c, _g in v), mean(g for _c, g in v))
               for _b, v in sorted(ish.items()) if _b >= 20.0]
    ls_rows = list(csv.DictReader(open(f"{out_dir}/off3-ls-curve.csv")))
    ls = {}
    for r in ls_rows:
        ls.setdefault(float(r["budget_s"]), []).append(
            (float(r["cpu_s"]), float(r["gap"])))
    ls_pts = [(mean(c for c, _g in v), mean(g for _c, g in v))
              for _b, v in sorted(ls.items())]
    inc_file = (f"{out_dir}/off4-enum-incremental3.csv"
                if os.path.exists(f"{out_dir}/off4-enum-incremental3.csv")
                else f"{out_dir}/off4-enum-incremental.csv")
    inc_rows = list(csv.DictReader(open(inc_file)))
    inc = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in inc_rows:
        rep = int(r["rep"]) if "rep" in r else 0
        inc[r["shell"]][int(r["member"])][r["instance"]][rep] = (
            float(r["cum_cpu_s"]), float(r["best_gap"]))
    inc_pts = {}
    for sh in ("twbf-18", "greedy-12", "bbf-288"):
        rows_k = []
        for k, per_inst in sorted(inc[sh].items()):
            # one aggregation everywhere: per-instance median over reps,
            # then mean over instances
            cpus = [median([c for c, _g in reps.values()])
                    for reps in per_inst.values()]
            gaps = [next(iter(reps.values()))[1] for reps in per_inst.values()]
            rows_k.append((mean(cpus), mean(gaps)))
        inc_pts[sh] = rows_k

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = [p[0] for p in dense_pts] + [p[0] for p in ish_pts]
    ys = [p[1] for p in dense_pts] + [p[1] for p in ish_pts]
    lo = [p[1] for p in dense_lo] + [p[1] for p in ish_pts]
    hi = [p[1] for p in dense_hi] + [p[1] for p in ish_pts]
    ax.fill_between(xs, lo, hi, color="tab:blue", alpha=0.15,
                    label="RandomLS seed range")
    ax.plot(xs, ys, "o-", color="tab:blue", lw=1.6, ms=4,
            label="RandomLS (ISH 2017), 5-seed mean")
    ax.plot([p[0] for p in ls_pts], [p[1] for p in ls_pts], "s--",
            color="tab:cyan", lw=1.0, ms=4,
            label="LS on concurrent-work core")
    for sh, color in (("twbf-18", "tab:green"), ("greedy-12", "tab:purple"),
                      ("bbf-288", "tab:red")):
        p = inc_pts[sh]
        ax.plot([x for x, _y in p], [y for _x, y in p], ":",
                color=color, lw=1.4, label=f"{sh} (incremental)")
    for m, (c, g) in base_pts.items():
        ax.plot(c, g, "s", ms=7)
        ax.annotate(m, (c, g), textcoords="offset points", xytext=(5, 5),
                    fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("CPU time per instance (s, log scale)")
    ax.set_ylabel("mean gap (%)")
    ax.set_title("Cost--quality plane, 104 instances (instance-count weighted)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig-off4-frontier.pdf")
    plt.close(fig)
    print(f"wrote {fig_dir}/fig-off4-frontier.pdf", file=sys.stderr)
    return 0


def cmd_figures_off4_extra(out_dir: str) -> int:
    """OFF-4 extra figures (user-approved 2026-10-08, Fig.8-style proposals):
    (c) per-family mean-gap lines for the robustness section (always);
    (a) ZDF scale tier, gap and CPU vs n (when off4-zdf-tier.csv exists);
    (b) industrial A tier, per-instance gap lines (when off4-industrial.csv
    exists).
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir).parent / "off4-cq-frontier" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lb_map = {r["instance"]: int(r["lb"])
              for r in csv.DictReader(open(f"{out_dir}/label-matrix.csv"))}

    def fam(n):
        return "BKW" if n.startswith("BKW") else ("NT" if n[0] in "NT"
                                                  else "C")

    # ---- (c) per-family means ----
    tim = list(csv.DictReader(open(f"{out_dir}/off4-timing.csv")))
    per_im: dict[str, dict[str, list]] = {}
    for r in tim:
        per_im.setdefault(r["method"], {}).setdefault(
            r["instance"], []).append(
                100.0 * (int(r["height"]) - lb_map[r["instance"]])
                / lb_map[r["instance"]])
    fam_gaps: dict[str, dict[str, float]] = {}

    def put(method, label, gaps_by_inst):
        fam_gaps[label] = {
            f: mean(g for i, g in gaps_by_inst.items() if fam(i) == f)
            for f in ("C", "BKW", "NT")}
        fam_gaps[label]["Avg."] = mean(gaps_by_inst.values())

    name_map = {"bf-TN": "bf-TN", "ish-core": "ish-core",
                "twbf-best18": "twbf-18", "bbf-best288": "bbf-288",
                "fh": "fh"}
    for m, label in name_map.items():
        put(m, label, {i: sorted(v)[1] for i, v in per_im[m].items()})
    pf = {r["instance"]: float(r["portfolio_gap"])
          for r in csv.DictReader(open(f"{out_dir}/off3-portfolio.csv"))}
    put("greedy-12", "greedy-12", pf)
    dense = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"{out_dir}/off4-ish-dense.csv")):
        dense[float(r["budget_s"])][r["instance"]].append(float(r["gap"]))
    for budget, label in ((0.02, "RandomLS@0.02s"), (0.10, "RandomLS@0.1s")):
        put(label, label, {i: mean(v) for i, v in dense[budget].items()})

    order = ["bf-TN", "ish-core", "twbf-18", "bbf-288", "greedy-12", "fh",
             "RandomLS@0.02s", "RandomLS@0.1s"]
    marks = ["s", "o", "^", "v", "D", "P", "X", "<"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:cyan", "tab:olive"]
    xs = list(range(4))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for m, mk, c in zip(order, marks, colors):
        ys = [fam_gaps[m][f] for f in ("C", "BKW", "NT", "Avg.")]
        ax.plot(xs, ys, marker=mk, ms=4.5, lw=1.2, color=c, label=m)
    ax.set_xticks(xs)
    ax.set_xticklabels(["C (21)", "BKW (13)", "NT (70)", "Avg. (104)"])
    ax.set_ylabel("mean gap (%)")
    ax.set_title("Mean gap by instance family (RandomLS = five-seed mean)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig-off4-families.pdf")
    plt.close(fig)
    print(f"wrote {fig_dir}/fig-off4-families.pdf", file=sys.stderr)

    # ---- (a) ZDF scale tier ----
    zdf_path = f"{out_dir}/off4-zdf-tier.csv"
    if os.path.exists(zdf_path):
        rows = list(csv.DictReader(open(zdf_path)))
        data: dict[str, list] = defaultdict(list)
        for r in rows:
            data[r["method"]].append((int(r["n"]), float(r["gap"]),
                                      float(r["cpu_s"])))
        zorder = ["bf-TN", "ish-core", "twbf-18", "greedy-12", "fh",
                  "RandomLS@1s", "RandomLS@5s", "RandomLS@20s", "bbf-288",
                  "rl-policy"]
        marks = ["s", "o", "^", "D", "P", "X", "<", ">", "v", "*"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
                  "tab:brown", "tab:cyan", "tab:olive", "tab:pink",
                  "tab:red", "tab:gray"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
        for m, mk, c in zip(zorder, marks, colors):
            if m not in data:
                continue
            pts = sorted(data[m])
            a1.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                    ms=4.5, lw=1.2, color=c, label=m)
            a2.plot([p[0] for p in pts], [max(p[2], 1e-3) for p in pts],
                    marker=mk, ms=4.5, lw=1.2, color=c, label=m)
        for a in (a1, a2):
            a.set_xscale("log")
            a.grid(True, which="both", alpha=0.25)
            a.set_xlabel("n (log scale)")
        a2.set_yscale("log")
        a1.set_ylabel("gap (%)")
        a2.set_ylabel("CPU s (log scale)")
        a1.set_title("ZDF scale tier: quality")
        a2.set_title("ZDF scale tier: cost")
        a1.legend(fontsize=6.0, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig-off4-zdf.pdf")
        plt.close(fig)
        print(f"wrote {fig_dir}/fig-off4-zdf.pdf", file=sys.stderr)
    else:
        print("skip fig-off4-zdf.pdf (no off4-zdf-tier.csv yet)",
              file=sys.stderr)

    # ---- (b) industrial A tier ----
    ind_path = f"{out_dir}/off4-industrial.csv"
    if os.path.exists(ind_path):
        rows = list(csv.DictReader(open(ind_path)))
        insts = sorted({(r["instance"], int(r["n"])) for r in rows},
                       key=lambda t: t[1])
        x_of = {name: k for k, (name, _n) in enumerate(insts)}
        data: dict[str, dict] = defaultdict(dict)
        for r in rows:
            data[r["method"]][x_of[r["instance"]]] = float(r["gap"])
        iorder = ["bf-TN", "ish-core", "twbf-18", "greedy-12", "fh",
                  "RandomLS@5s", "RandomLS@20s", "bbf-288", "rl-policy"]
        marks = ["s", "o", "^", "D", "P", "X", ">", "v", "*"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
                  "tab:brown", "tab:cyan", "tab:pink", "tab:red", "tab:gray"]
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for m, mk, c in zip(iorder, marks, colors):
            if m not in data:
                continue
            xs_i = sorted(data[m])
            ax.plot(xs_i, [data[m][x] for x in xs_i], marker=mk, ms=3.5,
                    lw=0.9, color=c, label=m)
        ax.set_xticks([x_of[n] for n, _ in insts])
        ax.set_xticklabels([n for n, _ in insts], rotation=90, fontsize=4.5)
        ax.set_ylabel("gap (%)")
        ax.set_title("Industrial A set (43 furniture instances, sorted by n)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.0, ncol=3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig-off4-industrial.pdf")
        plt.close(fig)
        print(f"wrote {fig_dir}/fig-off4-industrial.pdf", file=sys.stderr)
    else:
        print("skip fig-off4-industrial.pdf (no off4-industrial.csv yet)",
              file=sys.stderr)
    return 0


def cmd_figures_off3_controls(out_dir: str) -> int:
    """OFF-3 controls figure (user-approved 2026-10-08, proposal d):
    composition gain per arm with paired 95% CI, dev (filled) vs final
    (open, exploratory), two panels (TWBF-18 / BBF-288). All numbers
    recomputed from the CSVs (same math as cmd_controls_summary / round2).
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir).parent / "off3-learned-composer" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(BOOT_SEED)

    def ci(diffs):
        boot = sorted(mean(rng.choice(diffs) for _ in range(len(diffs)))
                      for _ in range(BOOTSTRAP))
        return mean(diffs), boot[250], boot[9750]

    # ---- dev set ----
    rows = list(csv.DictReader(open(f"{out_dir}/off3-controls.csv")))
    arm_gaps: dict[str, dict[str, list]] = {}
    for r in rows:
        arm_gaps.setdefault(r["arm"], {}).setdefault(
            r["instance"], []).append(float(r["gap"]))
    gaps = {arm: {i: mean(v) for i, v in per.items()}
            for arm, per in arm_gaps.items()}
    rl_rows = {r["instance"]: r for r in csv.DictReader(
        open(f"{out_dir}/rl-instance-gaps.csv")) if r["family"] != "PV"}
    gaps["rl-policy"] = {i: float(r["rl_gap"]) for i, r in rl_rows.items()}
    gaps["train-sel-fixed"] = {i: float(r["best_fixed"])
                               for i, r in rl_rows.items()}
    tim = [r for r in csv.DictReader(open(f"{out_dir}/off4-timing.csv"))
           if r["method"] == "bbf-best288" and r["instance"] == "BKW13"]
    lb13 = next(int(r["lb"]) for r in csv.DictReader(
        open(f"{out_dir}/label-matrix.csv")) if r["instance"] == "BKW13")
    shell: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(open("docs/stage3-baselines.csv")):
        if r["algorithm"] in ("twbf-best18", "bbf-best288"):
            shell.setdefault(r["algorithm"], {})[r["instance"]] = float(
                r["gap_pct"])
    shell["bbf-best288"]["BKW13"] = 100.0 * (int(tim[0]["height"]) - lb13) / lb13

    names = sorted(gaps["rl-policy"])
    arms = ["rl-policy", "rand-step", "freq-step", "stage-table",
            "train-sel-fixed"]
    dev: dict[str, dict[str, tuple]] = {}
    for algo, alabel in (("twbf-best18", "TWBF-18"),
                         ("bbf-best288", "BBF-288")):
        for arm in arms:
            combo = [min(shell[algo][n], gaps[arm][n]) for n in names]
            diffs = [shell[algo][n] - c for n, c in zip(names, combo)]
            dev.setdefault(alabel, {})[arm] = ci(diffs)

    # ---- final set (exploratory) ----
    r2 = list(csv.DictReader(open(f"{out_dir}/off3-round2.csv")))
    farm: dict[str, dict[str, list]] = {}
    for r in r2:
        farm.setdefault(r["arm"], {}).setdefault(
            r["instance"], []).append(float(r["gap"]))
    fgaps = {arm: {i: mean(v) for i, v in per.items()}
             for arm, per in farm.items()}
    fin = {r["instance"]: r
           for r in csv.DictReader(open(f"{out_dir}/off3-final.csv"))}
    fnames = sorted(fin)
    fshell = {"TWBF-18": {i: float(r["twbf18"]) for i, r in fin.items()},
              "BBF-288": {i: float(r["bbf288"]) for i, r in fin.items()}}
    frl = {"TWBF-18": {i: float(r["min_twbf_rl"]) for i, r in fin.items()},
           "BBF-288": {i: float(r["min_bbf_rl"]) for i, r in fin.items()}}
    final: dict[str, dict[str, tuple]] = {}
    for alabel in ("TWBF-18", "BBF-288"):
        for arm in ("rl-policy", "rand-step", "freq-step", "stage-table"):
            if arm == "rl-policy":
                diffs = [fshell[alabel][i] - frl[alabel][i] for i in fnames]
            else:
                diffs = [fshell[alabel][i]
                         - min(fshell[alabel][i], fgaps[arm][i])
                         for i in fnames]
            final.setdefault(alabel, {})[arm] = ci(diffs)

    labels = ["RL policy", "rand-step", "freq-step", "stage-table",
              "train-sel-fixed"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for ax, alabel in zip(axes, ("TWBF-18", "BBF-288")):
        for k, arm in enumerate(arms):
            color = "tab:red" if arm == "rl-policy" else "tab:gray"
            m, lo, hi = dev[alabel][arm]
            ax.errorbar(k - 0.12, m, yerr=[[m - lo], [hi - m]], fmt="o-",
                        ms=5, lw=1.0, capsize=2.5, color=color,
                        label="dev (pre-registered)" if k == 0 else None)
            if arm in final[alabel]:
                m, lo, hi = final[alabel][arm]
                ax.errorbar(k + 0.12, m, yerr=[[m - lo], [hi - m]], fmt="s",
                            ms=5, mfc="none", lw=1.0, capsize=2.5,
                            color=color,
                            label="final (exploratory)" if k == 0 else None)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_title(alabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("composition gain over shell (pp)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, fontsize=7, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(fig_dir / "fig-controls.pdf")
    plt.close(fig)
    # submission/fig is a symlink to ../fig, so the submission version
    # picks the figure up automatically
    print(f"wrote {fig_dir}/fig-controls.pdf", file=sys.stderr)
    return 0


def cmd_ish_dense(out_dir: str) -> int:
    """OFF-4 review round 1: dense short-budget checkpoints + multi-seed.

    RandomLS at budgets 0.02/0.05/0.10/0.15/0.20/1/5 s with 5 seeds
    (3000..3004). Answers: does greedy-12/twbf-18 stay on the frontier at
    0.10-0.18s, and how large is RandomLS's seed variance?
    """

    from ish.rls import random_ls

    budgets = (0.02, 0.05, 0.10, 0.15, 0.20, 1.0, 5.0)
    out_path = f"{out_dir}/off4-ish-dense.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["instance"], float(r["budget_s"]), int(r["seed"]))
                    for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "family", "budget_s", "seed",
                             "gap", "wall_s", "cpu_s"])
        for _g, path in _instance_paths():
            instance = load_instance(path)
            fam = _family(instance.name)
            lb = _lower_bound(instance)
            for budget in budgets:
                for seed in range(3000, 3005):
                    if (instance.name, budget, seed) in done:
                        continue
                    t0c = process_time()
                    best_h, _pl, elapsed = random_ls(
                        instance, time_limit=budget, seed=seed
                    )
                    writer.writerow([instance.name, fam, f"{budget:.2f}",
                                     seed,
                                     f"{100.0 * (best_h - lb) / lb:.4f}",
                                     f"{elapsed:.3f}",
                                     f"{process_time() - t0c:.3f}"])
                handle.flush()
            print(f"[ish-dense] {instance.name} done", file=sys.stderr,
                  flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_enum_incremental(out_dir: str) -> int:
    """OFF-4 review round 1: incremental best-of curves for the shells.

    Per instance, run each shell member in its canonical order and record
    (cumulative CPU, running-best gap) after each member — the honest
    'anytime' curves of TWBF-18, BBF-288, and the fold-fair greedy-12
    (per-fold member order from the CV convention). Round 2 change: three
    reps per member, cumulative median, so the prefix costs reconcile with
    the full-run medians in off4-timing.csv.
    """

    import bbf as bbf_pkg
    import twbf as twbf_pkg
    from bbf import all_combinations as bbf_combos
    from twbf import all_combinations as twbf_combos

    from .model import Config, Ordering, PlacementPolicy, Selection
    from .solver import solve_config

    _names, configs_full, matrix, _lbm = _load_matrix(out_dir)
    for r in csv.DictReader(open(f"{out_dir}/label-matrix-pv.csv")):
        matrix.setdefault(r["instance"], {})
        if isinstance(matrix[r["instance"]], list):
            continue
        matrix[r["instance"]][r["config"]] = float(r["gap_pct"])

    def yvec(name):
        v = matrix[name]
        return [v[c] for c in configs_full] if isinstance(v, dict) else list(v)

    pf_by_fold = {}
    for held in ("C", "BKW", "NT"):
        train = [i for i in matrix if _family(i) != held]
        Y = {i: yvec(i) for i in train}
        pf_by_fold[held] = _greedy_portfolio(list(Y), Y, k=12)

    out_path = f"{out_dir}/off4-enum-incremental3.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["instance"], r["shell"], int(r["member"]),
                     int(r["rep"])) for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "family", "shell", "member", "rep",
                             "cum_cpu_s", "best_gap"])
        for _g, path in _instance_paths():
            instance = load_instance(path)
            fam = _family(instance.name)
            lb = _lower_bound(instance)
            for rep in range(3):
                # twbf-18 in the package's canonical enumeration order
                cum, best_h = 0.0, None
                for mi, combo in enumerate(twbf_combos()):
                    t0 = process_time()
                    sol = twbf_pkg.solve_combination(instance, combo,
                                                     validate=False)
                    cum += process_time() - t0
                    best_h = sol.height if best_h is None else min(best_h,
                                                                   sol.height)
                    if (instance.name, "twbf-18", mi, rep) not in done:
                        writer.writerow([instance.name, fam, "twbf-18", mi,
                                         rep, f"{cum:.4f}",
                                         f"{100.0 * (best_h - lb) / lb:.4f}"])
                # bbf-288 in the package's canonical enumeration order
                cum, best_h = 0.0, None
                for mi, combo in enumerate(bbf_combos()):
                    t0 = process_time()
                    sol = bbf_pkg.solve_combination(instance, combo,
                                                    validate=False)
                    cum += process_time() - t0
                    best_h = sol.height if best_h is None else min(best_h,
                                                                   sol.height)
                    if (instance.name, "bbf-288", mi, rep) not in done:
                        writer.writerow([instance.name, fam, "bbf-288", mi,
                                         rep, f"{cum:.4f}",
                                         f"{100.0 * (best_h - lb) / lb:.4f}"])
                # greedy-12 in selection order (fold-fair membership)
                cum, best_h = 0.0, None
                for mi, c in enumerate(pf_by_fold[fam]):
                    o, s, rest = configs_full[c].split("/")
                    p, tw = rest.split("+tw")[0], rest.split("+tw")[1][0]
                    vn = rest.split("+vn")[1][0]
                    cfg = Config(
                        ordering=Ordering(o), selection=Selection(s),
                        placement=PlacementPolicy(p), tower_removal=(tw == "T"),
                        vertical_niche=(vn == "T"),
                    )
                    t0 = process_time()
                    _pl, sky, _ = solve_config(instance, cfg)
                    cum += process_time() - t0
                    h2 = max(sky)
                    best_h = h2 if best_h is None else min(best_h, h2)
                    if (instance.name, "greedy-12", mi, rep) not in done:
                        writer.writerow([instance.name, fam, "greedy-12", mi,
                                         rep, f"{cum:.4f}",
                                         f"{100.0 * (best_h - lb) / lb:.4f}"])
                handle.flush()
            print(f"[enum-inc3] {instance.name} done", file=sys.stderr,
                  flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_report_off4(out_dir: str) -> int:
    """OFF-4 single source of truth: generate off4-tables.tex + numbers JSON.

    All paper tables and headline numbers are computed here from the raw
    CSVs (no hand-typed numbers in main.tex for these tables).
    """

    import json

    out_tex = Path(out_dir).parent / "off4-cq-frontier" / "off4-tables.tex"
    out_tab1 = Path(out_dir).parent / "off4-cq-frontier" / "off4-tab-timing.tex"
    out_tab2 = Path(out_dir).parent / "off4-cq-frontier" / "off4-tab-curve.tex"
    out_json = Path(out_dir).parent / "off4-cq-frontier" / "off4-numbers.json"

    lb_map = {}
    for r in csv.DictReader(open(f"{out_dir}/label-matrix.csv")):
        lb_map[r["instance"]] = int(r["lb"])

    def fam(n):
        return "BKW" if n.startswith("BKW") else ("NT" if n[0] in "NT"
                                                  else "C")

    def fam_gap(rows_gaps):  # instance-count weighted over the 104
        return mean(rows_gaps)

    # ---- static methods from off4-timing (+ rl-1t + portfolio)
    tim = list(csv.DictReader(open(f"{out_dir}/off4-timing.csv")))
    per_im: dict[str, dict[str, list]] = {}
    for r in tim:
        per_im.setdefault(r["method"], {}).setdefault(r["instance"], []).append(
            (float(r["cpu_s"]),
             100.0 * (int(r["height"]) - lb_map[r["instance"]]) / lb_map[r["instance"]])
        )
    pf = list(csv.DictReader(open(f"{out_dir}/off3-portfolio.csv")))
    rl1 = list(csv.DictReader(open(f"{out_dir}/off4-timing-rl-1t.csv")))

    methods: dict[str, dict] = {}

    def add_method(name, layer, per_inst):
        # per_inst: instance -> (cpu, gap) single measurement
        gaps = {i: g for i, (c, g) in per_inst.items()}
        cpus = {i: c for i, (c, g) in per_inst.items()}
        methods[name] = {
            "layer": layer,
            "gap": fam_gap(list(gaps.values())),
            "gap_by_family": {f: mean(g for i, g in gaps.items()
                                      if fam(i) == f) for f in ("C", "BKW", "NT")},
            "gap_family_equal": mean(mean(g for i, g in gaps.items()
                                          if fam(i) == f)
                                     for f in ("C", "BKW", "NT")),
            "cpu_sum": sum(cpus.values()),
            "cpu_inst": mean(cpus.values()),
            "cpu_p95": sorted(cpus.values())[int(0.95 * len(cpus)) - 1],
            "gaps": gaps, "cpus": cpus,
        }

    for m in ("bf-TN", "ish-core", "twbf-best18", "bbf-best288", "fh"):
        per_inst = {}
        for inst, rows in per_im[m].items():
            rows.sort()
            per_inst[inst] = rows[1]
        layer = {"bf-TN": "constructive", "ish-core": "constructive",
                 "twbf-best18": "enumeration", "bbf-best288": "enumeration",
                 "fh": "search"}[m]
        add_method({"twbf-best18": "twbf-18",
                    "bbf-best288": "bbf-288"}.get(m, m), layer, per_inst)
    add_method("greedy-12", "enumeration",
               {r["instance"]: (float(r["cpu_s"]), float(r["portfolio_gap"]))
                for r in pf})
    rl_cpu_by_inst: dict[str, list[float]] = {}
    for r in rl1:
        rl_cpu_by_inst.setdefault(r["instance"], []).append(
            float(r["cpu_s"]))
    rl_gap_by_inst = {
        r["instance"]: 100.0 * (int(r["height"]) - lb_map[r["instance"]])
        / lb_map[r["instance"]]
        for r in rl1 if r["rep"] == "0"
    }
    add_method("rl-policy (capped)", "learned",
               {i: (sorted(v)[1], rl_gap_by_inst[i])
                for i, v in rl_cpu_by_inst.items()})
    # the bbf+RL combo runs the shell then the policy as two serial runs,
    # so its cost is the sum of the two per-instance costs; the shell's cost
    # uses the same incremental continuous runs as the prefix curve
    bbf_end_cpu = {}
    for r in csv.DictReader(open(f"{out_dir}/off4-enum-incremental3.csv")):
        if r["shell"] == "bbf-288" and int(r["member"]) == 287:
            bbf_end_cpu.setdefault(r["instance"], []).append(
                float(r["cum_cpu_s"]))
    bbf_end_cpu = {i: median(v) for i, v in bbf_end_cpu.items()}
    bbf_rl = {}
    for r in rl1:
        if r["rep"] != "0":
            continue
        i = r["instance"]
        bb = per_im["bbf-best288"][i]
        bb.sort()
        bbf_rl[i] = (bbf_end_cpu[i] + methods["rl-policy (capped)"]["cpus"][i],
                     min(bb[1][1],
                         100.0 * (int(r["height"]) - lb_map[i]) / lb_map[i]))
    add_method("bbf-288+RL (capped)", "learned+shell", bbf_rl)

    # ---- RandomLS: original curve (seed 3000) + dense 5-seed
    ish = defaultdict(list)
    for r in csv.DictReader(open(f"{out_dir}/off4-ish-curve.csv")):
        ish[float(r["budget_s"])].append(
            (r["instance"], float(r["gap"]), float(r["cpu_s"]),
             float(r["wall_s"])))
    dense = defaultdict(list)
    for r in csv.DictReader(open(f"{out_dir}/off4-ish-dense.csv")):
        dense[(float(r["budget_s"]), int(r["seed"]))].append(
            (r["instance"], float(r["gap"]), float(r["cpu_s"]),
             float(r["wall_s"])))

    curve_rows = []
    for b in sorted(ish):
        rows = ish[b]
        curve_rows.append({
            "budget": b,
            "gap": mean(g for _i, g, _c, _w in rows),
            "cpu": mean(c for _i, _g, c, _w in rows),
            "wall": mean(w for _i, _g, _c, w in rows),
            "lb_stops": sum(1 for _i, g, _c, _w in rows if g == 0.0),
            "gap_by_family": {f: mean(g for i, g, _c, _w in rows
                                      if fam(i) == f)
                              for f in ("C", "BKW", "NT")},
        })
    dense_rows = []
    for b in sorted({b for b, _s in dense}):
        per_seed = [mean(g for _i, g, _c, _w in dense[(b, s)])
                    for s in sorted({s for _b, s in dense if _b == b})]
        cpu_vals = [c for (bb, _s), rows in dense.items() if bb == b
                    for _i, _g, c, _w in rows]
        wall_vals = [w for (bb, _s), rows in dense.items() if bb == b
                     for _i, _g, _c, w in rows]
        stops = sum(1 for (bb, _s), rows in dense.items() if bb == b
                    for _i, g, _c, _w in rows if g == 0.0)
        dense_rows.append({"budget": b, "gap": mean(per_seed),
                           "sd": pstdev_(per_seed), "cpu": mean(cpu_vals),
                           "wall": mean(wall_vals),
                           "lb_stops": stops / 5.0,
                           "seeds": per_seed})

    # ---- incremental curves (per member k: median cum cpu / mean best gap)
    inc_file = (f"{out_dir}/off4-enum-incremental3.csv"
                if os.path.exists(f"{out_dir}/off4-enum-incremental3.csv")
                else f"{out_dir}/off4-enum-incremental.csv")
    inc = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in csv.DictReader(open(inc_file)):
        rep = int(r["rep"]) if "rep" in r else 0
        inc[r["shell"]][int(r["member"])][r["instance"]][rep] = (
            float(r["cum_cpu_s"]), float(r["best_gap"]))
    incr = {}
    for sh, ms in inc.items():
        rows_k = []
        for k, per_inst in sorted(ms.items()):
            # one aggregation everywhere: per-instance median over reps,
            # then mean over instances
            cpus = [median([c for c, _g in reps.values()])
                    for reps in per_inst.values()]
            gaps = [next(iter(reps.values()))[1] for reps in per_inst.values()]
            rows_k.append((k, mean(cpus), mean(gaps)))
        incr[sh] = rows_k

    # ---- frontier (mean plane), dense checkpoints included
    points = [(m, d["cpu_inst"], d["gap"]) for m, d in methods.items()]
    dense_by_b = {r["budget"]: r for r in dense_rows}
    curve_pts = []
    for r in curve_rows:
        b = r["budget"]
        if b in dense_by_b:
            curve_pts.append((f"RandomLS@{b:g}s", dense_by_b[b]["cpu"],
                              dense_by_b[b]["gap"]))
        else:
            curve_pts.append((f"RandomLS@{b:g}s (1 seed)", r["cpu"],
                              r["gap"]))
    for r in dense_rows:
        if r["budget"] in ish:
            continue
        curve_pts.append((f"RandomLS@{r['budget']:g}s", r["cpu"],
                          r["gap"]))
    allpts = sorted(points + curve_pts, key=lambda t: t[1])
    # shell prefixes are legitimate candidate methods: include them
    for sh, rows_k in incr.items():
        for k, c, g in rows_k:
            allpts.append((f"{sh}#{k + 1}", c, g))
    allpts.sort(key=lambda t: t[1])
    frontier = []
    best = float("inf")
    for name, c, g in allpts:
        if g < best - 1e-12:
            frontier.append((name, c, g))
            best = g

    numbers = {
        "methods": {m: {k: v for k, v in d.items() if k != "gaps"
                        and k != "cpus"}
                    for m, d in methods.items()},
        "curve": curve_rows,
        "dense": dense_rows,
        "frontier": [list(t) for t in frontier],
    }
    with open(out_json, "w") as fh:
        json.dump(numbers, fh, indent=1)

    # ---- off4-tables.tex
    def row(name, layer, d):
        return (f"{name} & {layer} & {d['gap']:.3f} & "
                f"{d['cpu_sum']:.2f} & {d['cpu_inst']:.3f} \\\\")
    # ---- write the two tables as separate inputs (single source of truth)
    # Shell costs in the main table come from the incremental continuous
    # runs' final member (same runs as the prefix curves), so table and
    # figure share one data source; bf/ish/fh/rl keep their off4-timing
    # / off4-timing-rl-1t median-of-3 costs.
    tab1 = ["% auto-generated by cch.off2_papers report-off4; do not edit",
            "\\begin{tabular}{lcccc}", "\\toprule",
            "method & layer & gap & CPU sum (s) & CPU/instance (s) \\\\",
            "\\midrule"]
    order = ["bf-TN", "ish-core", "twbf-18", "greedy-12", "bbf-288", "fh",
             "rl-policy (capped)", "bbf-288+RL (capped)"]
    frontier_names = {n for n, _c, _g in frontier}
    for m in order:
        d = methods[m]
        if m in ("twbf-18", "bbf-288", "greedy-12"):
            k_end, c_end, _g_end = incr[{
                "twbf-18": "twbf-18", "bbf-288": "bbf-288",
                "greedy-12": "greedy-12"}[m]][-1]
            d = dict(d)
            d["cpu_sum"], d["cpu_inst"] = c_end * 104, c_end
        name = f"\\textbf{{{m}}}" if m in frontier_names else m
        tab1.append(row(name, d["layer"], d))
    # truncated-portfolio prefix points (frontier members)
    for k, c, g in incr["greedy-12"]:
        if k + 1 in (2, 3):
            name = f"greedy-12 prefix-{k + 1}"
            if f"greedy-12#{k + 1}" in frontier_names:
                name = f"\\textbf{{{name}}}"
            tab1.append(f"{name} & enum.\\ prefix & {g:.3f} & "
                        f"{c * 104:.1f} & {c:.3f} \\\\")
    for r in curve_rows:
        if r["budget"] in (0.2, 1.0, 5.0):
            b = r["budget"]
            d = dense_by_b.get(b)
            name = f"RandomLS@{b:g}s"
            gap_v = d["gap"] if d else r["gap"]
            cpu_v = d["cpu"] if d else r["cpu"]
            if name in frontier_names:
                name = f"\\textbf{{{name}}}"
            tab1.append(
                f"{name} & search & {gap_v:.3f} & "
                f"{cpu_v * 104:.1f} & {cpu_v:.3f} \\\\")
    for r in dense_rows:
        if r["budget"] in (0.02, 0.05):
            name = f"RandomLS@{r['budget']:g}s"
            if name in frontier_names:
                name = f"\\textbf{{{name}}}"
            tab1.append(
                f"{name} & search & {r['gap']:.3f} & "
                f"{r['cpu'] * 104:.1f} & {r['cpu']:.3f} \\\\")
    tab1 += ["\\bottomrule", "\\end{tabular}"]
    with open(out_tab1, "w") as fh:
        fh.write("\n".join(tab1) + "\n")

    tab2 = ["% auto-generated; do not edit", "\\begin{tabular}{lccccc}",
            "\\toprule",
            "budget/inst & mean gap & wall/inst & cpu/inst & LB stops"
            " & seed sd \\\\", "\\midrule"]
    dense_by_b = {r["budget"]: r for r in dense_rows}
    for r in curve_rows:
        b = r["budget"]
        if b in dense_by_b:
            d = dense_by_b[b]
            tab2.append(
                f"{b:g}\\,s & {d['gap']:.3f} & {r['wall']:.3f} & "
                f"{d['cpu']:.3f} & {int(r['lb_stops'])}/104 & "
                f"{d['sd']:.3f} (5 seeds) \\\\")
        else:
            tab2.append(
                f"{b:g}\\,s & {r['gap']:.3f} & {r['wall']:.3f} & "
                f"{r['cpu']:.3f} & {int(r['lb_stops'])}/104 & 1 seed \\\\")
    for r in dense_rows:
        if r["budget"] in ish:
            continue
        tab2.append(
            f"{r['budget']:g}\\,s & {r['gap']:.3f} & {r['wall']:.3f} & "
            f"{r['cpu']:.3f} & {r['lb_stops']:.0f}/104 & "
            f"{r['sd']:.3f} (5 seeds) \\\\")
    tab2 += ["\\bottomrule", "\\end{tabular}"]
    with open(out_tab2, "w") as fh:
        fh.write("\n".join(tab2) + "\n")
    print(f"wrote {out_tab1}, {out_tab2}, {out_json}", file=sys.stderr)
    print("frontier:", " -> ".join(f"{n}({c:.3f}s,{g:.2f})"
                                   for n, c, g in frontier), flush=True)
    return 0


def pstdev_(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def cmd_controls_summary(out_dir: str) -> int:
    """Compose the control arms into the published shells and report CIs."""

    rows = list(csv.DictReader(open(f"{out_dir}/off3-controls.csv")))
    arm_gaps: dict[str, dict[str, list]] = {}
    for r in rows:
        arm_gaps.setdefault(r["arm"], {}).setdefault(
            r["instance"], []).append(float(r["gap"]))
    gaps = {
        arm: {i: mean(v) for i, v in per.items()}
        for arm, per in arm_gaps.items()
    }
    rl_rows = {
        r["instance"]: r for r in csv.DictReader(
            open(f"{out_dir}/rl-instance-gaps.csv")) if r["family"] != "PV"
    }
    gaps["rl-policy"] = {i: float(r["rl_gap"]) for i, r in rl_rows.items()}
    gaps["instance-level"] = {i: float(r["best_fixed"])
                              for i, r in rl_rows.items()}
    tim = [r for r in csv.DictReader(open(f"{out_dir}/off4-timing.csv"))
           if r["method"] == "bbf-best288" and r["instance"] == "BKW13"]
    lb13 = next(int(r["lb"]) for r in csv.DictReader(
        open(f"{out_dir}/label-matrix.csv")) if r["instance"] == "BKW13")
    shell: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(open("docs/stage3-baselines.csv")):
        if r["algorithm"] in ("twbf-best18", "bbf-best288"):
            shell.setdefault(r["algorithm"], {})[r["instance"]] = float(
                r["gap_pct"]
            )
    shell["bbf-best288"]["BKW13"] = 100.0 * (int(tim[0]["height"]) - lb13) / lb13

    rng = random.Random(BOOT_SEED)

    def ci(diffs):
        boot = sorted(mean(rng.choice(diffs) for _ in range(len(diffs)))
                      for _ in range(BOOTSTRAP))
        return mean(diffs), boot[250], boot[9750]

    names = sorted(gaps["rl-policy"])
    print(f"{'arm':16s} {'alone':>7s} | {'shell':>10s} {'combo':>7s} "
          f"{'gain':>7s} {'CI':>18s}")
    for arm in ("rl-policy", "rand-step", "freq-step", "stage-table",
                "instance-level"):
        alone = mean(gaps[arm][n] for n in names)
        for algo in ("twbf-best18", "bbf-best288"):
            combo = [min(shell[algo][n], gaps[arm][n]) for n in names]
            diffs = [shell[algo][n] - c for n, c in zip(names, combo)]
            m, lo, hi = ci(diffs)
            print(f"{arm:16s} {alone:7.3f} | {algo:10s} "
                  f"{mean(combo):7.3f} {m:+7.3f} [{lo:+.3f},{hi:+.3f}]",
                  flush=True)
    return 0


def cmd_zdf_tier(out_dir: str) -> int:
    """OFF-4 scale tier (user-approved 2026-10-08): representative methods on
    ZDF large instances (n 580..10064), including the DEPLOYED greedy-12
    (the OFF-1 global portfolio selected on all 104) and RandomLS at
    1/5/20s. RL policy and BBF-288 only up to n=2532 (cost caps, recorded).
    """

    from .model import Config, Ordering, PlacementPolicy, Selection
    from .solver import solve_config
    from .shell import _sort_key

    instances = []
    for z in ("zdf1", "zdf4", "zdf6", "zdf8", "zdf10", "zdf12"):
        path = f"data/zdf/{z}.ins2D"
        instances.append((z.upper(), path))
    _names, configs_full, matrix, _lbm = _load_matrix(out_dir)
    from .off2_learn import _portfolio_mask
    deployed = _portfolio_mask(configs_full)
    from ish.solver import best_fit_pack
    from ish.rls import SORT_RULES, random_ls
    import fh as fh_pkg

    def run_config(instance, c):
        o, s, rest = configs_full[c].split("/")
        p, tw = rest.split("+tw")[0], rest.split("+tw")[1][0]
        vn = rest.split("+vn")[1][0]
        cfg = Config(ordering=Ordering(o), selection=Selection(s),
                     placement=PlacementPolicy(p), tower_removal=(tw == "T"),
                     vertical_niche=(vn == "T"))
        _pl, sky, _ = solve_config(instance, cfg)
        return max(sky)

    out_path = f"{out_dir}/off4-zdf-tier.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["instance"], r["method"]) for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "n", "method", "gap", "cpu_s",
                             "wall_s"])
        for name, path in instances:
            instance = load_instance(path)
            n = len(instance.items)
            lb = _lower_bound(instance)
            lb_area = -(-sum(it.area for it in instance.items)
                        // instance.strip_width)

            def rec(method, h, cpu, wall):
                if (name, method) in done:
                    return
                writer.writerow([name, n, method,
                                 f"{100.0 * (h - lb) / lb:.4f}",
                                 f"{cpu:.3f}", f"{wall:.3f}"])
                handle.flush()

            def timed(fn):
                t0c, t0w = process_time(), perf_counter()
                h = fn()
                return h, process_time() - t0c, perf_counter() - t0w

            rec("bf-TN", *timed(lambda: run_config(
                instance,
                [c for c, s in enumerate(configs_full)
                 if s == "width/widest-fit/TN+twT+vnF"][0])))
            rec("ish-core", *timed(lambda: max(
                best_fit_pack(sorted(list(instance.items), key=k),
                              instance.strip_width)[1]
                for _nm, k in SORT_RULES)))
            rec("twbf-18", *timed(lambda: __import__("twbf").solve(
                instance)[0].height))
            rec("greedy-12", *timed(lambda: min(
                run_config(instance, c) for c in deployed)))
            fh_budget = None if n <= 250 else (5000 if n <= 500 else 500)
            rec("fh", *timed(lambda: fh_pkg.fast_heuristic(
                list(instance.items), instance.strip_width,
                swap_budget=fh_budget)[1]))
            for budget in (1.0, 5.0, 20.0):
                rec(f"RandomLS@{budget:g}s", *timed(lambda b=budget: random_ls(
                    instance, time_limit=b, seed=3000)[0]))
            if n <= 2532:
                rec("bbf-288", *timed(lambda: __import__("bbf").solve(
                    instance)[0].height))
                models = pickle.load(
                    open(f"{out_dir}/rl-models-final.pkl", "rb"))
                policy = _greedy_policy(models)
                actions, configs = _action_set(False)
                rec("rl-policy", *timed(lambda: _episode(
                    instance, lb, policy, 0, actions, configs)[2]))
            print(f"[zdf] {name} (n={n}) done, LB={lb} (area {lb_area})",
                  file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_industrial_a(out_dir: str) -> int:
    """OFF-4 industrial tier (user-approved 2026-10-08): the same
    representative-method protocol as zdf-tier on the 43 furniture-industry
    instances (data/industrial-a/, Macedo et al. "A" set from 2DPackLib).
    All n<=809, so bbf-288 and rl-policy run on every instance.
    """

    from .model import Config, Ordering, PlacementPolicy, Selection
    from .solver import solve_config
    from .shell import _sort_key

    root = "data/industrial-a"
    names = sorted((f"A{i}" for i in range(1, 44)),
                   key=lambda s: int(s[1:]))
    instances = [(name, f"{root}/{name}.ins2D") for name in names]
    _names, configs_full, matrix, _lbm = _load_matrix(out_dir)
    from .off2_learn import _portfolio_mask
    deployed = _portfolio_mask(configs_full)
    from ish.solver import best_fit_pack
    from ish.rls import SORT_RULES, random_ls
    import fh as fh_pkg

    def run_config(instance, c):
        o, s, rest = configs_full[c].split("/")
        p, tw = rest.split("+tw")[0], rest.split("+tw")[1][0]
        vn = rest.split("+vn")[1][0]
        cfg = Config(ordering=Ordering(o), selection=Selection(s),
                     placement=PlacementPolicy(p), tower_removal=(tw == "T"),
                     vertical_niche=(vn == "T"))
        _pl, sky, _ = solve_config(instance, cfg)
        return max(sky)

    out_path = f"{out_dir}/off4-industrial.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["instance"], r["method"]) for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "n", "method", "gap", "cpu_s",
                             "wall_s"])
        for name, path in instances:
            instance = load_instance(path)
            n = len(instance.items)
            lb = _lower_bound(instance)

            def rec(method, h, cpu, wall):
                if (name, method) in done:
                    return
                writer.writerow([name, n, method,
                                 f"{100.0 * (h - lb) / lb:.4f}",
                                 f"{cpu:.3f}", f"{wall:.3f}"])
                handle.flush()

            def timed(fn):
                t0c, t0w = process_time(), perf_counter()
                h = fn()
                return h, process_time() - t0c, perf_counter() - t0w

            rec("bf-TN", *timed(lambda: run_config(
                instance,
                [c for c, s in enumerate(configs_full)
                 if s == "width/widest-fit/TN+twT+vnF"][0])))
            rec("ish-core", *timed(lambda: max(
                best_fit_pack(sorted(list(instance.items), key=k),
                              instance.strip_width)[1]
                for _nm, k in SORT_RULES)))
            rec("twbf-18", *timed(lambda: __import__("twbf").solve(
                instance)[0].height))
            rec("greedy-12", *timed(lambda: min(
                run_config(instance, c) for c in deployed)))
            fh_budget = None if n <= 250 else (5000 if n <= 500 else 500)
            rec("fh", *timed(lambda: fh_pkg.fast_heuristic(
                list(instance.items), instance.strip_width,
                swap_budget=fh_budget)[1]))
            for budget in (1.0, 5.0, 20.0):
                rec(f"RandomLS@{budget:g}s", *timed(lambda b=budget: random_ls(
                    instance, time_limit=b, seed=3000)[0]))
            rec("bbf-288", *timed(lambda: __import__("bbf").solve(
                instance)[0].height))
            models = pickle.load(
                open(f"{out_dir}/rl-models-final.pkl", "rb"))
            policy = _greedy_policy(models)
            actions, configs = _action_set(False)
            rec("rl-policy", *timed(lambda: _episode(
                instance, lb, policy, 0, actions, configs)[2]))
            print(f"[ind-a] {name} (n={n}) done, LB={lb}",
                  file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_budget(out_dir: str) -> int:
    """OFF-3 revision step 3: equal-CPU-budget controls (external review #6).

    Per instance, budget = the RL member's median CPU time from the
    idle-window timing run. Arms: ``budget-rand`` (repeated uniform-random
    constructions, best-of) and ``budget-ls`` (first-improvement single-swap
    local search from the perimeter-sorted start; stage2-best core
    perimeter/fitness-number/SN + TR via the off2_state sequence path).
    Budgets are enforced with process_time, so the comparison is CPU-fair
    regardless of machine load; overshoot is at most one pack per arm.
    """

    from .model import Config, Ordering, PlacementPolicy, Selection
    from .off2_state import solve_config_traced
    from .shell import _sort_key

    per: dict[str, list[float]] = {}
    for r in csv.DictReader(open(f"{out_dir}/off4-timing.csv")):
        if r["method"] == "rl-policy":
            per.setdefault(r["instance"], []).append(float(r["cpu_s"]))
    rl_cpu = {i: sorted(v)[1] for i, v in per.items()}

    config = Config(
        ordering=Ordering.PERIMETER, selection=Selection.FITNESS_NUMBER,
        placement=PlacementPolicy.SN, tower_removal=True,
    )
    actions, configs = _action_set(False)

    def pack_seq(instance, items):
        _p, skyline, _m, _t = solve_config_traced(instance, config,
                                                  sequence=items)
        return max(skyline)

    out_path = f"{out_dir}/off3-budget.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {r["instance"] for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "family", "budget_s",
                             "rand_gap", "rand_packs", "rand_cpu",
                             "ls_gap", "ls_packs", "ls_cpu"])
        for _g, path in _instance_paths():
            instance = load_instance(path)
            if instance.name in done:
                continue
            fam = _family(instance.name)
            lb = _lower_bound(instance)
            budget = rl_cpu[instance.name]

            # arm 1: best-of repeated uniform-random constructions
            best_h, tries = None, 0
            t0 = process_time()
            while True:
                policy = lambda _f, r: r.randrange(len(actions))
                _t, _hc, h_tr = _episode(
                    instance, lb, policy, 2000 + tries, actions, configs
                )
                tries += 1
                if best_h is None or h_tr < best_h:
                    best_h = h_tr
                if process_time() - t0 >= budget or best_h == lb:
                    break
            rand_cpu = process_time() - t0

            # arm 2: first-improvement single-swap local search
            items = list(instance.items)
            rng = random.Random(3000)
            current = sorted(items,
                             key=lambda it: _sort_key(Ordering.PERIMETER, it))
            t0 = process_time()
            cur_h = best_ls = pack_seq(instance, current)
            packs = 1
            while process_time() - t0 < budget and best_ls > lb:
                a, b = rng.sample(range(len(items)), 2)
                cand = list(current)
                cand[a], cand[b] = cand[b], cand[a]
                h = pack_seq(instance, cand)
                packs += 1
                if h <= cur_h:
                    current, cur_h = cand, h
                    if h < best_ls:
                        best_ls = h
            ls_cpu = process_time() - t0

            writer.writerow([
                instance.name, fam, f"{budget:.4f}",
                f"{100.0 * (best_h - lb) / lb:.4f}", tries,
                f"{rand_cpu:.4f}",
                f"{100.0 * (best_ls - lb) / lb:.4f}", packs,
                f"{ls_cpu:.4f}",
            ])
            handle.flush()
            print(f"[budget] {instance.name} done (b={budget:.2f}s, "
                  f"rand {tries} packs, ls {packs} packs)",
                  file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


PROFILE_INSTANCES = ("c1-p1.ins2D", "c4-p2.ins2D", "c7-p1.ins2D",
                     "BKW05.ins2D", "BKW13.ins2D")
LS_CURVE_BUDGETS = (0.2, 1.0, 5.0, 20.0, 60.0)


def cmd_profile(out_dir: str) -> int:
    """Inference-tax decomposition of the RL rollout (external review #6).

    Splits per-instance rollout CPU into feature computation, the nine
    per-step model predicts, the engine step, and tower removal.
    """

    from . import off2_rl as rl
    from .model import PlacementPolicy

    out_path = f"{out_dir}/off3-profile.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "n", "cpu_total_s", "t_features_s",
                         "t_predict_s", "t_engine_s", "t_tr_s", "steps"])
        for rel in PROFILE_INSTANCES:
            path = next(p for _g, p in _instance_paths()
                        if p.endswith(rel))
            instance = load_instance(path)
            fam = _family(instance.name)
            with open(f"{out_dir}/rl-models-{fam}.pkl", "rb") as fh:
                models = pickle.load(fh)
            policy = _greedy_policy(models)
            lb = _lower_bound(instance)
            actions, configs = _action_set(False)
            st = rl._State(instance)
            rng = random.Random(0)
            t_feat = t_pred = t_engine = 0.0
            steps = 0
            t_start = process_time()
            while st.remaining:
                t0 = process_time()
                feats = rl._features(st, instance, lb)
                t_feat += process_time() - t0
                t0 = process_time()
                a = policy(feats, rng)
                t_pred += process_time() - t0
                sel, pol = actions[a]
                t0 = process_time()
                rl._step(st, configs[(sel, pol)], sel)
                t_engine += process_time() - t0
                steps += 1
            t0 = process_time()
            _pl, sky_tr, _m = rl._reduce_towers(
                list(st.placements), list(st.skyline), PlacementPolicy.LM
            )
            t_tr = process_time() - t0
            total = process_time() - t_start
            writer.writerow([instance.name, len(instance.items),
                             f"{total:.3f}", f"{t_feat:.3f}",
                             f"{t_pred:.3f}", f"{t_engine:.3f}",
                             f"{t_tr:.3f}", steps])
            handle.flush()
            print(f"[profile] {instance.name}: total {total:.2f}s "
                  f"(predict {100*t_pred/total:.0f}%)", file=sys.stderr,
                  flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_ls_curve(out_dir: str) -> int:
    """Multi-budget local-search curve (0.2/1/5/20/60s per instance).

    Same LS semantics as the budget arm (perimeter start, first-improvement
    single swap, stage2-best core + TR, LB early stop, process_time caps).
    Feeds the OFF-4 frontier and the small-budget counterfactual for OFF-3.
    """

    from .model import Config, Ordering, PlacementPolicy, Selection
    from .off2_state import solve_config_traced
    from .shell import _sort_key

    config = Config(
        ordering=Ordering.PERIMETER, selection=Selection.FITNESS_NUMBER,
        placement=PlacementPolicy.SN, tower_removal=True,
    )

    def pack_seq(instance, items):
        _p, skyline, _m, _t = solve_config_traced(instance, config,
                                                  sequence=items)
        return max(skyline)

    out_path = f"{out_dir}/off3-ls-curve.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["instance"], float(r["budget_s"]))
                    for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "family", "budget_s", "gap",
                             "packs", "cpu_s"])
        for _g, path in _instance_paths():
            instance = load_instance(path)
            fam = _family(instance.name)
            lb = _lower_bound(instance)
            for budget in LS_CURVE_BUDGETS:
                if (instance.name, budget) in done:
                    continue
                items = list(instance.items)
                rng = random.Random(3000)
                current = sorted(
                    items, key=lambda it: _sort_key(Ordering.PERIMETER, it)
                )
                t0 = process_time()
                cur_h = best_h = pack_seq(instance, current)
                packs = 1
                while process_time() - t0 < budget and best_h > lb:
                    a, b = rng.sample(range(len(items)), 2)
                    cand = list(current)
                    cand[a], cand[b] = cand[b], cand[a]
                    h = pack_seq(instance, cand)
                    packs += 1
                    if h <= cur_h:
                        current, cur_h = cand, h
                        if h < best_h:
                            best_h = h
                cpu_s = process_time() - t0
                writer.writerow([instance.name, fam, f"{budget:.1f}",
                                 f"{100.0 * (best_h - lb) / lb:.4f}",
                                 packs, f"{cpu_s:.3f}"])
                handle.flush()
            print(f"[ls-curve] {instance.name} done", file=sys.stderr,
                  flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_timing_rl(out_dir: str) -> int:
    """Thread-clean RL rollout timing (rl-policy-1t), 3 reps, serial.

    The idle-window off4-timing.csv ran the main process WITHOUT thread
    caps, so rl-policy's 2160s was ~99% OpenMP team-creation churn per
    single-row predict (profiling: 430ms -> 0.88ms per call, 488x).
    This subcommand re-measures with OMP/BLAS threads = 1.
    """

    actions, configs = _action_set(False)
    models_by_fold = {}
    for held in FOLDS:
        with open(f"{out_dir}/rl-models-{held}.pkl", "rb") as handle:
            models_by_fold[held] = pickle.load(handle)
    out_path = f"{out_dir}/off4-timing-rl-1t.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["instance"], int(r["rep"]))
                    for r in csv.DictReader(handle)}
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["instance", "family", "method", "rep",
                             "cpu_s", "wall_s", "height"])
        for _g, path in _instance_paths():
            instance = load_instance(path)
            fam = _family(instance.name)
            lb = _lower_bound(instance)
            policy = _greedy_policy(models_by_fold[fam])
            for rep in range(3):
                if (instance.name, rep) in done:
                    continue
                t0c, t0w = process_time(), perf_counter()
                _trans, _hc, h_tr = _episode(
                    instance, lb, policy, 0, actions, configs
                )
                writer.writerow([instance.name, fam, "rl-policy-1t", rep,
                                 f"{process_time() - t0c:.4f}",
                                 f"{perf_counter() - t0w:.4f}", h_tr])
                handle.flush()
            print(f"[timing-rl] {instance.name}", file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0



    """Same-budget verdict: shells + budget arms vs shells + RL."""

    rows = list(csv.DictReader(open(f"{out_dir}/off3-budget.csv")))
    gaps = {
        "budget-rand": {r["instance"]: float(r["rand_gap"]) for r in rows},
        "budget-ls": {r["instance"]: float(r["ls_gap"]) for r in rows},
    }
    rl_rows = {
        r["instance"]: r for r in csv.DictReader(
            open(f"{out_dir}/rl-instance-gaps.csv")) if r["family"] != "PV"
    }
    gaps["rl-policy"] = {i: float(r["rl_gap"]) for i, r in rl_rows.items()}
    tim = [r for r in csv.DictReader(open(f"{out_dir}/off4-timing.csv"))
           if r["method"] == "bbf-best288" and r["instance"] == "BKW13"]
    lb13 = next(int(r["lb"]) for r in csv.DictReader(
        open(f"{out_dir}/label-matrix.csv")) if r["instance"] == "BKW13")
    shell: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(open("docs/stage3-baselines.csv")):
        if r["algorithm"] in ("twbf-best18", "bbf-best288"):
            shell.setdefault(r["algorithm"], {})[r["instance"]] = float(
                r["gap_pct"]
            )
    shell["bbf-best288"]["BKW13"] = 100.0 * (int(tim[0]["height"]) - lb13) / lb13

    rng = random.Random(BOOT_SEED)

    def ci(diffs):
        boot = sorted(mean(rng.choice(diffs) for _ in range(len(diffs)))
                      for _ in range(BOOTSTRAP))
        return mean(diffs), boot[250], boot[9750]

    names = sorted(gaps["rl-policy"])
    print(f"{'arm':12s} {'alone':>7s} {'packs':>6s} | per-shell combo gains")
    for arm in ("rl-policy", "budget-rand", "budget-ls"):
        alone = mean(gaps[arm][n] for n in names)
        packs = ""
        if arm != "rl-policy":
            col = "rand_packs" if arm == "budget-rand" else "ls_packs"
            packs = f"{mean(int(r[col]) for r in rows):6.0f}"
        line = f"{arm:12s} {alone:7.3f} {packs:>6s} |"
        for algo in ("twbf-best18", "bbf-best288"):
            combo = [min(shell[algo][n], gaps[arm][n]) for n in names]
            diffs = [shell[algo][n] - c for n, c in zip(names, combo)]
            m, lo, hi = ci(diffs)
            line += f"  {algo}: {mean(combo):.3f} gain {m:+.3f} [{lo:+.3f},{hi:+.3f}]"
        print(line, flush=True)
    return 0



    """Compose the control arms into the published shells and report CIs."""

    rows = list(csv.DictReader(open(f"{out_dir}/off3-controls.csv")))
    arm_gaps: dict[str, dict[str, list]] = {}
    for r in rows:
        arm_gaps.setdefault(r["arm"], {}).setdefault(
            r["instance"], []).append(float(r["gap"]))
    gaps = {
        arm: {i: mean(v) for i, v in per.items()}
        for arm, per in arm_gaps.items()
    }
    rl_rows = {
        r["instance"]: r for r in csv.DictReader(
            open(f"{out_dir}/rl-instance-gaps.csv")) if r["family"] != "PV"
    }
    gaps["rl-policy"] = {i: float(r["rl_gap"]) for i, r in rl_rows.items()}
    gaps["instance-level"] = {i: float(r["best_fixed"])
                              for i, r in rl_rows.items()}
    tim = [r for r in csv.DictReader(open(f"{out_dir}/off4-timing.csv"))
           if r["method"] == "bbf-best288" and r["instance"] == "BKW13"]
    lb13 = next(int(r["lb"]) for r in csv.DictReader(
        open(f"{out_dir}/label-matrix.csv")) if r["instance"] == "BKW13")
    shell: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(open("docs/stage3-baselines.csv")):
        if r["algorithm"] in ("twbf-best18", "bbf-best288"):
            shell.setdefault(r["algorithm"], {})[r["instance"]] = float(
                r["gap_pct"]
            )
    shell["bbf-best288"]["BKW13"] = 100.0 * (int(tim[0]["height"]) - lb13) / lb13

    rng = random.Random(BOOT_SEED)

    def ci(diffs):
        boot = sorted(mean(rng.choice(diffs) for _ in range(len(diffs)))
                      for _ in range(BOOTSTRAP))
        return mean(diffs), boot[250], boot[9750]

    names = sorted(gaps["rl-policy"])
    print(f"{'arm':16s} {'alone':>7s} | {'shell':>10s} {'combo':>7s} "
          f"{'gain':>7s} {'CI':>18s}")
    for arm in ("rl-policy", "rand-step", "freq-step", "stage-table",
                "instance-level"):
        alone = mean(gaps[arm][n] for n in names)
        for algo in ("twbf-best18", "bbf-best288"):
            combo = [min(shell[algo][n], gaps[arm][n]) for n in names]
            diffs = [shell[algo][n] - c for n, c in zip(names, combo)]
            m, lo, hi = ci(diffs)
            print(f"{arm:16s} {alone:7.3f} | {algo:10s} "
                  f"{mean(combo):7.3f} {m:+7.3f} [{lo:+.3f},{hi:+.3f}]",
                  flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cch.off2_papers", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rl-instances", "combo", "timing", "controls",
                 "controls-summary", "budget", "budget-summary",
                 "profile", "ls-curve", "timing-rl", "multiseed",
                 "final-eval", "figures", "figures-off4", "round2",
                 "round3", "ish-curve", "ish-dense", "enum-incremental",
                 "report-off4", "zdf-tier", "industrial-a",
                 "figures-off4-extra", "figures-off3-controls"):
        child = sub.add_parser(name)
        child.add_argument("--out-dir", default="docs/off2-learn")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.command == "rl-instances":
        return cmd_rl_instances(args.out_dir)
    if args.command == "combo":
        return cmd_combo(args.out_dir)
    if args.command == "timing":
        return cmd_timing(args.out_dir)
    if args.command == "controls":
        return cmd_controls(args.out_dir)
    if args.command == "controls-summary":
        return cmd_controls_summary(args.out_dir)
    if args.command == "budget":
        return cmd_budget(args.out_dir)
    if args.command == "budget-summary":
        return cmd_budget_summary(args.out_dir)
    if args.command == "profile":
        return cmd_profile(args.out_dir)
    if args.command == "ls-curve":
        return cmd_ls_curve(args.out_dir)
    if args.command == "timing-rl":
        return cmd_timing_rl(args.out_dir)
    if args.command == "multiseed":
        return cmd_multiseed(args.out_dir)
    if args.command == "final-eval":
        return cmd_final_eval(args.out_dir)
    if args.command == "figures":
        return cmd_figures(args.out_dir)
    if args.command == "figures-off4":
        return cmd_figures_off4(args.out_dir)
    if args.command == "round2":
        return cmd_round2(args.out_dir)
    if args.command == "round3":
        return cmd_round3(args.out_dir)
    if args.command == "ish-curve":
        return cmd_ish_curve(args.out_dir)
    if args.command == "ish-dense":
        return cmd_ish_dense(args.out_dir)
    if args.command == "enum-incremental":
        return cmd_enum_incremental(args.out_dir)
    if args.command == "report-off4":
        return cmd_report_off4(args.out_dir)
    if args.command == "zdf-tier":
        return cmd_zdf_tier(args.out_dir)
    if args.command == "industrial-a":
        return cmd_industrial_a(args.out_dir)
    if args.command == "figures-off4-extra":
        return cmd_figures_off4_extra(args.out_dir)
    if args.command == "figures-off3-controls":
        return cmd_figures_off3_controls(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
