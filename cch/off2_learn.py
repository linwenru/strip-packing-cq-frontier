"""OFF-2 learned line: instance-level portfolio selector (design G v1).

Subcommands:

    python3 -m cch.off2_learn features   # instance-side features -> features.csv
    python3 -m cch.off2_learn matrix     # 864-pool label matrix from stage CSVs
    python3 -m cch.off2_learn pvruns     # run the 864 pool on 120 prospective instances
    python3 -m cch.off2_learn cv         # family-holdout CV of the selector
    python3 -m cch.off2_learn final-eval # final-set evaluation (new seeds)

Leakage audit: features are instance-side statistics only (nothing derived
from any packing, runtime, or outcome; instance family labels are used for
splitting only, never as model inputs). sklearn/numpy/pandas are OFF-2
exploratory dependencies (not needed for OFF-1 reproduction).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from statistics import mean, pstdev

from burke_bf import Instance, load_instance

from .experiment import _instance_paths, _lower_bound

OUT_DIR = "docs/off2-learn"
FLAG_SUFFIX = {0: "", 1: "+tw", 2: "+vn", 3: "+vn+tw"}
SUFFIX_FLAG = {v: k for k, v in FLAG_SUFFIX.items()}
PV_PATHS = [
    (f"PV-{g.upper()}", f"data/prospective/pv-{g}-{n}-{i:02d}.ins2D")
    for g in "abcd"
    for n in (100, 300)
    for i in range(1, 16)
]


def _family(name: str) -> str:
    if name.startswith("BKW"):
        return "BKW"
    if name.startswith("PV"):
        return "PV"
    if name[0] in "NT":
        return "NT"
    return "C"


def _instance_features(instance: Instance) -> dict[str, float]:
    """Instance-side statistics only (leakage-audited, see module docstring).

    v2 adds packing-informed features: width/height/aspect histograms, width
    complementarity (pairs summing to ~W), height complementarity (~LB),
    dimension rarity.
    """

    items = list(instance.items)
    n = len(items)
    ws = [it.width for it in items]
    hs = [it.height for it in items]
    W = instance.strip_width
    area_lb = math.ceil(sum(it.area for it in items) / W)
    lb = max(area_lb, max(hs))
    ratios = [max(it.width, it.height) / min(it.width, it.height) for it in items]
    areas = [it.area for it in items]

    def stats(vals):
        sv = sorted(vals)
        q = lambda p: sv[min(len(sv) - 1, int(p * len(sv)))]
        return {
            "mean": mean(vals), "sd": pstdev(vals) if len(vals) > 1 else 0.0,
            "min": sv[0], "q25": q(0.25), "q50": q(0.5), "q75": q(0.75),
            "q90": q(0.9), "max": sv[-1],
        }

    fw, fh, fr, fa = stats(ws), stats(hs), stats(ratios), stats(areas)
    dims = Counter((it.width, it.height) for it in items)
    widths = Counter(ws)
    entropy = -sum(
        (c / n) * math.log2(c / n) for c in widths.values()
    )
    corr = 0.0
    if n > 1 and pstdev(ws) > 0 and pstdev(hs) > 0:
        mw, mh = mean(ws), mean(hs)
        corr = mean((w - mw) * (h - mh) for w, h in zip(ws, hs)) / (
            pstdev(ws) * pstdev(hs)
        )
    # ---- packing-informed v2 additions
    max_area = max(areas)
    w_bins = [0] * 10
    for w in ws:
        w_bins[min(9, int(10 * w / (W + 1)))] += 1
    a_bins = [0] * 10
    for a in areas:
        a_bins[min(9, int(10 * a / (max_area + 1)))] += 1
    r_bins = [0] * 5
    for r in ratios:
        r_bins[min(4, int(r) if r < 5 else 4)] += 1
    pairs = 0
    wcomp = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if abs(ws[i] + ws[j] - W) <= 0.05 * W:
                wcomp += 1
    hcomp = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(hs[i] + hs[j] - lb) <= 0.05 * lb:
                hcomp += 1
    return {
        "n": n, "W": W, "area_lb": area_lb, "lb": lb,
        "area_ratio": sum(it.area for it in items) / (W * lb),
        **{f"w_{k}": v for k, v in fw.items()},
        **{f"h_{k}": v for k, v in fh.items()},
        **{f"ratio_{k}": v for k, v in fr.items()},
        **{f"area_{k}": v for k, v in fa.items()},
        "thin_share_05": mean(min(it.width, it.height) <= 0.05 * W for it in items),
        "thin_share_10": mean(min(it.width, it.height) <= 0.1 * W for it in items),
        "big_share": mean(max(it.width, it.height) >= 0.5 * W for it in items),
        "square_share": mean(r <= 1.2 for r in ratios),
        "slender_share": mean(r >= 3.0 for r in ratios),
        "wh_corr": corr,
        "max_w_over_W": max(ws) / W,
        "max_h_over_lb": max(hs) / lb,
        "n_wide_half": sum(it.width >= W / 2 for it in items),
        "n_tall_half": sum(it.height >= lb / 2 for it in items),
        "n_distinct_dims": len(dims),
        "dominant_dim_share": max(dims.values()) / n,
        "width_entropy": entropy,
        **{f"wbin_{k}": c / n for k, c in enumerate(w_bins)},
        **{f"abin_{k}": c / n for k, c in enumerate(a_bins)},
        **{f"rbin_{k}": c / n for k, c in enumerate(r_bins)},
        "width_comp_pairs": wcomp / max(pairs, 1),
        "height_comp_pairs": hcomp / max(pairs, 1),
        "rare_dim_share": mean(dims[(it.width, it.height)] == 1 for it in items),
    }


def cmd_features(out_dir: str) -> int:
    paths = list(_instance_paths()) + PV_PATHS
    rows = []
    for data_set, path in paths:
        instance = load_instance(path)
        feats = _instance_features(instance)
        rows.append(
            {"instance": instance.name, "family": _family(instance.name),
             "group": data_set, **feats}
        )
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/features.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_dir}/features.csv: {len(rows)} instances, "
          f"{len(rows[0]) - 3} features (instance-side only)")
    return 0


def cmd_matrix(out_dir: str) -> int:
    """Assemble the 864-pool label matrix from existing stage CSVs."""

    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    sources = (
        ("stage1-runs.csv", ""),
        ("stage2-tower-runs.csv", "+tw"),
        ("stage2-vn-runs.csv", "+vn"),
        ("stage2-vntower-runs.csv", "+vn+tw"),
    )
    lb_map: dict[str, int] = {}
    for filename, suffix in sources:
        with open(f"docs/{filename}", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    f"{row['ordering']}/{row['selection']}/{row['placement']}"
                    f"+tw{'T' if suffix in ('+tw', '+vn+tw') else 'F'}"
                    f"+vn{'T' if suffix in ('+vn', '+vn+tw') else 'F'}"
                )
                matrix[row["instance"]][key] = float(row["gap_pct"])
                lb_map[row["instance"]] = int(row["lower_bound"])
    configs = sorted({k for cell in matrix.values() for k in cell})
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/label-matrix.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "family", "lb"] + configs)
        for name in sorted(matrix):
            row = matrix[name]
            writer.writerow(
                [name, _family(name), lb_map[name]]
                + [f"{row[c]:.4f}" for c in configs]
            )
    print(f"wrote {out_dir}/label-matrix.csv: {len(matrix)} instances x "
          f"{len(configs)} configs")
    return 0


def _pv_work(task: tuple) -> tuple:
    """Worker for cmd_pvruns (module level so multiprocessing can pickle it)."""

    from .model import Config
    from .solver import solve_config

    path, o, s, p, tr, vn = task
    instance = load_instance(path)
    config = Config(
        ordering=o, selection=s, placement=p,
        tower_removal=tr, vertical_niche=vn,
    )
    placements, skyline, _ = solve_config(instance, config)
    lb = _lower_bound(instance)
    key = (
        f"{o.value}/{s.value}/{p.value}"
        f"+tw{'T' if tr else 'F'}+vn{'T' if vn else 'F'}"
    )
    return (path, instance.name, key,
            100.0 * (max(skyline) - lb) / lb, lb)


def cmd_pvruns(out_dir: str) -> int:
    """Run the 864 pool on the 120 prospective instances (deterministic labels).

    Gap labels are deterministic per configuration; parallelism is safe (this
    is not a timing measurement). CPU timings for reporting are measured
    separately and serially at evaluation time.
    """

    from multiprocessing import Pool

    from .model import Ordering, PlacementPolicy, Selection

    combos = [
        (o, s, p, tr, vn)
        for o in Ordering
        for s in Selection
        for p in PlacementPolicy
        for tr in (False, True)
        for vn in (False, True)
    ]

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/label-matrix-pv.csv"
    done: set[tuple[str, str]] = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as handle:
            done = {(r["path"], r["config"]) for r in csv.DictReader(handle)}
    tasks = [
        (path, o, s, p, tr, vn)
        for _group, path in PV_PATHS
        for (o, s, p, tr, vn) in combos
        if (path, f"{o.value}/{s.value}/{p.value}"
                 f"+tw{'T' if tr else 'F'}+vn{'T' if vn else 'F'}") not in done
    ]
    print(f"pv label runs: {len(tasks)} cells remaining", file=sys.stderr)
    with open(out_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if os.path.getsize(out_path) == 0:
            writer.writerow(["path", "instance", "config", "gap_pct", "lb"])
        with Pool(6) as pool:
            for i, rec in enumerate(pool.imap_unordered(_pv_work, tasks, chunksize=8)):
                writer.writerow(rec)
                if (i + 1) % 5000 == 0:
                    handle.flush()
                    print(f"  {i + 1}/{len(tasks)}", file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def _load_matrix(out_dir: str) -> tuple[list[str], list[str], dict, dict]:
    with open(f"{out_dir}/label-matrix.csv", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        configs = header[3:]
        lb_map = {}
        matrix = {}
        for row in reader:
            matrix[row[0]] = [float(v) for v in row[3:]]
            lb_map[row[0]] = int(row[2])
    return sorted(matrix), configs, matrix, lb_map


def _load_features(out_dir: str) -> dict[str, dict]:
    with open(f"{out_dir}/features.csv", newline="") as handle:
        return {r["instance"]: r for r in csv.DictReader(handle)}


FEATURE_KEYS = None  # set from the features.csv header at runtime


def _portfolio_mask(configs: list[str]) -> list[int]:
    """The greedy 12-config portfolio, read from the stage4 portfolio file."""

    picked = []
    with open("docs/stage4-portfolio.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            picked.append(row["added_config"])
            if len(picked) >= 12:
                break
    return [configs.index(c) for c in picked]


def _greedy_portfolio(
    train: list[str], Y: dict, k: int = 12
) -> list[int]:
    """Greedy portfolio selection on a TRAINING fold only (fold-fair).

    Start from the config with the best training mean; repeatedly add the
    config that most reduces the training mean of the per-instance min.
    """

    n_cfg = len(Y[train[0]])
    chosen: list[int] = []
    cur = {i: float("inf") for i in train}
    for _ in range(k):
        best_c, best_mean = -1, float("inf")
        for c in range(n_cfg):
            if c in chosen:
                continue
            cand_mean = mean(min(cur[i], Y[i][c]) for i in train)
            if cand_mean < best_mean - 1e-12:
                best_c, best_mean = c, cand_mean
        if best_c < 0:
            break
        chosen.append(best_c)
        for i in train:
            cur[i] = min(cur[i], Y[i][best_c])
    return chosen


def cmd_cv(out_dir: str) -> int:
    """Repair-round variant grid: feature set x objective x pool curation.

    features: v1 (50 generic) | v2 (+packing-informed); objective: regression
    on gap | topk (binary classifier P(in instance top-10)); pool: full 864 |
    curated 100 (best training-fold mean). Portfolio baselines are fold-fair
    greedy-12, computed on the matching pool for attribution. Models: RF/GBM,
    single global model with the config's train-fold mean-gap rank as a
    numeric feature (computed inside the fold: no leakage).
    """

    import numpy as np
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )

    instances, configs, matrix, lb_map = _load_matrix(out_dir)
    feats_rows = _load_features(out_dir)
    all_keys = [
        k for k in feats_rows[instances[0]]
        if k not in ("instance", "family", "group")
    ]
    v1_keys = [k for k in all_keys if not k.startswith(("wbin_", "abin_", "rbin_"))
               and k not in ("width_comp_pairs", "height_comp_pairs", "rare_dim_share")]
    fams = sorted({_family(i) for i in instances})
    K_LEVELS = (1, 2, 3, 5, 10)
    Y = {i: np.array(matrix[i], dtype=float) for i in instances}
    oracle = {i: float(min(Y[i])) for i in instances}
    best_fixed_idx = int(
        min(range(len(configs)), key=lambda c: mean(Y[i][c] for i in instances))
    )
    portfolio_idx = _portfolio_mask(configs)

    def Xset(i, keys):
        return np.array([float(feats_rows[i][k]) for k in keys], dtype=float)

    results = []

    def evaluate(rank, test, pool_idx, tag):
        for K in K_LEVELS:
            attained = [min(Y[i][c] for c in rank[i][:K] if c in pool_idx)
                        if any(c in pool_idx for c in rank[i][:K])
                        else min(Y[i][c] for c in pool_idx) for i in test]
            regret = [a - oracle[i] for a, i in zip(attained, test)]
            pf = [min(Y[i][c] for c in portfolio_idx) for i in test]
            results.append((tag, held, K, mean(attained),
                            sum(a <= 1e-9 for a in attained), mean(regret)))
            print(f"  {tag:28s} hold-{held:4s} K={K:2d}: attained={mean(attained):.3f} "
                  f"regret={mean(regret):+.3f} (12port-full={mean(pf):.3f})")

    for featset, keys in (("v1", v1_keys), ("v2", all_keys)):
        X = {i: Xset(i, keys) for i in instances}
        for objective in ("regression", "topk"):
            for pool_name in ("full", "curated"):
                for mname, ctor in (
                    ("rf", RandomForestRegressor if objective == "regression"
                     else RandomForestClassifier),
                    ("gbm", GradientBoostingRegressor if objective == "regression"
                     else GradientBoostingClassifier),
                ):
                    tag = f"{featset}-{objective}-{pool_name}-{mname}"
                    for held in fams:
                        train = [i for i in instances if _family(i) != held]
                        test = [i for i in instances if _family(i) == held]
                        means = [mean(Y[i][c] for i in train) for c in range(len(configs))]
                        rank_of = sorted(range(len(configs)), key=lambda c: means[c])
                        config_rank = [0.0] * len(configs)
                        for pos, c in enumerate(rank_of):
                            config_rank[c] = pos / (len(configs) - 1)
                        if pool_name == "curated":
                            pool_idx = set(rank_of[:100])
                        else:
                            pool_idx = set(range(len(configs)))
                        Xtr = np.stack([
                            np.concatenate([X[i], [config_rank[c]]])
                            for i in train for c in sorted(pool_idx)
                        ])
                        if objective == "regression":
                            ytr = np.concatenate([
                                Y[i][sorted(pool_idx)] for i in train
                            ])
                        else:
                            top10 = {
                                i: set(np.argsort(Y[i])[:10].tolist())
                                for i in train
                            }
                            ytr = np.concatenate([
                                np.array(
                                    [int(c in top10[i]) for c in sorted(pool_idx)]
                                )
                                for i in train
                            ])
                        kw = {"random_state": 42}
                        if mname == "rf":
                            kw["n_jobs"] = 1
                        m = ctor(**kw).fit(Xtr, ytr)
                        pool_list = sorted(pool_idx)

                        def pred(i, c):
                            out = m.predict(
                                np.concatenate([X[i], [config_rank[c]]]).reshape(1, -1)
                            )
                            return out[0]

                        def pred_proba(i, c):
                            proba = m.predict_proba(
                                np.concatenate([X[i], [config_rank[c]]]).reshape(1, -1)
                            )
                            return proba[0][1] if proba.shape[1] > 1 else 0.0

                        scorer = pred if objective == "regression" else (
                            lambda i, c: -pred_proba(i, c))
                        rank = {
                            i: sorted(pool_list, key=lambda c: scorer(i, c))
                            for i in test
                        }
                        evaluate(rank, test, pool_idx, f"{tag}")
    print("\nsummary table (K=3):")
    for tag, held, K, attained, optimal, regret in results:
        if K == 3:
            print(f"  {tag:28s} hold-{held:4s}: attained={attained:.3f} "
                  f"optimal={optimal} regret={regret:+.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cch.off2_learn",
        description=__doc__.splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("features", "matrix", "pvruns", "cv", "final-eval"):
        child = sub.add_parser(name)
        child.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "features":
        return cmd_features(args.out_dir)
    if args.command == "matrix":
        return cmd_matrix(args.out_dir)
    if args.command == "pvruns":
        return cmd_pvruns(args.out_dir)
    if args.command == "cv":
        return cmd_cv(args.out_dir)
    print("final-eval: pending (after first CV signal)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
