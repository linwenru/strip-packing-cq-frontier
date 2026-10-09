"""OFF-2 RL v4: Double DQN + Dueling function-class falsification arm.

Design: docs/off2-learn/design-rl.md v4 (user "不愿放弃", 2026-09-28).
Only the Q-function class changes relative to the v3 main arm — neural TD
(DDQN target decoupling, dueling V/A streams) instead of tree-ensemble
batch regression. The 9 narrow actions, 15 features, behavior episodes
(reused from rl-episodes-v3-main.csv), 3 eps-greedy collection rounds,
4 family-holdout folds, baselines, budget executor and the hard gate are
all unchanged, so the arms are directly comparable.

    python3 -m cch.off2_dqn cv   # folds {C, BKW, NT, PV}
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import mean
from time import process_time

from burke_bf import load_instance

from .experiment import _instance_paths, _lower_bound
from .off2_learn import _family
from .off2_rl import (
    CPU_CAP_S,
    EPS_GREEDY_EPISODES,
    GEN_PV_HOLD_SLICE,
    N_WORKERS,
    _action_set,
    _episode,
    _eval_fold,
    _gen_many,
    _load_episodes,
)

HIDDEN = 128
GRAD_STEPS_INIT = 30_000
GRAD_STEPS_ROUND = 10_000
TARGET_SYNC = 2_000
BATCH = 4096
LR = 3e-4
COLLECT_ROUNDS = 3
N_FEAT = 15


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_net(n_feat: int, n_actions: int):
    import torch.nn as nn

    class DuelingQ(nn.Module):
        """Dueling Q-network: shared body, V(s) + centred A(s,a) streams."""

        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(n_feat, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            )
            self.value = nn.Linear(HIDDEN, 1)
            self.advantage = nn.Linear(HIDDEN, n_actions)

        def forward(self, x):
            h = self.body(x)
            adv = self.advantage(h)
            return self.value(h) + adv - adv.mean(dim=1, keepdim=True)

    return DuelingQ()


def _episodes_to_arrays(episodes: dict, n_feat: int):
    import numpy as np

    X, A, R, Xn, D = [], [], [], [], []
    for _key, ep in episodes.items():
        for feats, a, r, next_feats, done in ep:
            X.append(feats[:n_feat])
            A.append(a)
            R.append(r)
            Xn.append((next_feats if next_feats is not None else feats)[:n_feat])
            D.append(done)
    return (
        np.array(X, dtype=np.float32), np.array(A, dtype=np.int64),
        np.array(R, dtype=np.float32), np.array(Xn, dtype=np.float32),
        np.array(D, dtype=np.float32),
    )


def _train_dqn(arrays, n_actions: int, steps: int, seed: int, net=None):
    """DDQN training on the transition arrays; returns the CPU-side net."""

    import numpy as np
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device()
    X, A, R, Xn, D = arrays
    if net is None:
        net = _build_net(X.shape[1], n_actions)
    online = net.to(device)
    target = _build_net(X.shape[1], n_actions).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=LR)
    huber = torch.nn.HuberLoss()
    Xt = torch.as_tensor(X, device=device)
    At = torch.as_tensor(A, device=device)
    Rt = torch.as_tensor(R, device=device)
    Xnt = torch.as_tensor(Xn, device=device)
    Dt = torch.as_tensor(D, device=device)
    n = len(Rt)
    losses = []
    online.train()
    for step in range(1, steps + 1):
        ix = torch.as_tensor(rng.integers(0, n, BATCH), device=device)
        q = online(Xt[ix]).gather(1, At[ix].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # DDQN: select with the online net, evaluate with the target net
            next_a = online(Xnt[ix]).argmax(dim=1, keepdim=True)
            next_q = target(Xnt[ix]).gather(1, next_a).squeeze(1)
            y = Rt[ix] + (1.0 - Dt[ix]) * next_q
        loss = huber(q, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % TARGET_SYNC == 0:
            target.load_state_dict(online.state_dict())
        if step % 500 == 0 or step == steps:
            losses.append(float(loss))
    return online.to("cpu"), mean(losses[-10:]) if losses else float("nan")


def _greedy_nn_policy(net):
    import torch

    torch.set_num_threads(1)
    net = net.to("cpu").eval()

    def policy(feats, _rng):
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            return int(net(x).argmax(dim=1).item())

    return policy


_DQN_CTX: dict = {}


def _dqn_eps_init(state_dict, n_actions, wide, mech, macro_m):
    """Pool initializer: restore the frozen Q-net once per worker."""

    import torch

    torch.set_num_threads(1)
    net = _build_net(N_FEAT, n_actions)
    net.load_state_dict(state_dict)
    net.eval()
    _DQN_CTX["net"] = net
    _DQN_CTX["actions"], _DQN_CTX["configs"] = _action_set(wide)
    _DQN_CTX["mech"] = mech
    _DQN_CTX["macro_m"] = macro_m


def _dqn_eps_one(task):
    """Pool worker: one eps-greedy episode with the NN policy."""

    import torch

    path, ep = task
    instance = load_instance(path)
    lb = _lower_bound(instance)
    net = _DQN_CTX["net"]
    actions = _DQN_CTX["actions"]

    def policy(feats, rng):
        if rng.random() < 0.2:
            return rng.randrange(len(actions))
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            return int(net(x).argmax(dim=1).item())

    transitions, _hc, _ht = _episode(
        instance, lb, policy, 1000 + ep, actions, _DQN_CTX["configs"],
        mech=_DQN_CTX["mech"], macro_m=_DQN_CTX["macro_m"],
    )
    return instance.name, f"eg{ep}", transitions


def _cv(out_dir: str) -> int:
    """Family-holdout CV of the DDQN arm (design-rl v4: identical protocol)."""

    base_specs = list(_instance_paths()) + [
        (f"PV-{g.upper()}", f"data/prospective/pv-{g}-{n}-{i:02d}.ins2D")
        for g in "abcd" for n in (100, 300) for i in range(1, 16)
    ]
    gen_specs, _final = _gen_many(out_dir)
    family_map: dict[str, str] = {}
    gen_index: dict[str, int] = {}
    for _g, p in base_specs:
        family_map[load_instance(p).name] = _family(load_instance(p).name)
    for _g, p in gen_specs:
        name = load_instance(p).name
        family_map[name] = "GEN"
        gen_index[name] = int(Path(p).stem.rsplit("-", 1)[1])

    actions, configs = _action_set(False)
    ep_path = f"{out_dir}/rl-episodes-v3-main.csv"
    print(f"loading behavior episodes from {ep_path}", file=sys.stderr)
    episodes = _load_episodes(ep_path)
    print(f"device: {_device()}", file=sys.stderr, flush=True)

    started_cpu = process_time()
    for held in ("C", "BKW", "NT", "PV"):
        if process_time() - started_cpu > CPU_CAP_S:
            print(f"CPU cap reached before dqn/{held}; truncating",
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
        print(f"[dqn] fold hold-{held}: train episodes {len(train_eps)}",
              file=sys.stderr, flush=True)
        arrays = _episodes_to_arrays(train_eps, N_FEAT)
        net, loss = _train_dqn(arrays, len(actions), GRAD_STEPS_INIT, 42)
        print(f"  init train {GRAD_STEPS_INIT} steps, huber~{loss:.6f}",
              file=sys.stderr, flush=True)
        for round_ix in range(COLLECT_ROUNDS):
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
            with Pool(N_WORKERS, initializer=_dqn_eps_init,
                      initargs=(net.state_dict(), len(actions), False,
                                False, 1)) as pool:
                for name, ep_key, transitions in pool.imap_unordered(
                    _dqn_eps_one, eg_tasks, chunksize=4
                ):
                    eps_extra[(name, ep_key)] = transitions
            # deterministic insertion order (see design-rl v4 addendum)
            for key in sorted(eps_extra):
                train_eps[key] = eps_extra[key]
            arrays = _episodes_to_arrays(train_eps, N_FEAT)
            net, loss = _train_dqn(arrays, len(actions), GRAD_STEPS_ROUND,
                                   42 + round_ix + 1, net=net)
            print(f"  round {round_ix + 1}/{COLLECT_ROUNDS}: "
                  f"+{len(eg_tasks)} eps, huber~{loss:.6f}",
                  file=sys.stderr, flush=True)
        policy = _greedy_nn_policy(net)
        results = _eval_fold(out_dir, held, base_specs, policy, actions,
                             configs, dict(wide=False, mech=False, macro_m=1))
        tag = f"dqn-hold-{held}"
        print(f"  {tag}: RL={results['rl']:.3f} portfolio={results['portfolio']:.3f} "
              f"best_fixed={results['best_fixed']:.3f} oracle={results['oracle']:.3f} "
              f"fallback={results['fallback']:.2f}", flush=True)
        cv_path = f"{out_dir}/rl-cv-dqn.csv"
        with open(cv_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if os.path.getsize(cv_path) == 0:
                writer.writerow(["tag", "variant", "held", "rl", "portfolio",
                                 "best_fixed", "oracle", "fallback"])
            writer.writerow([tag, "dqn", held, f"{results['rl']:.4f}",
                             f"{results['portfolio']:.4f}",
                             f"{results['best_fixed']:.4f}",
                             f"{results['oracle']:.4f}",
                             f"{results['fallback']:.4f}"])
    print(f"wrote {out_dir}/rl-cv-dqn.csv", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cch.off2_dqn", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("cv")
    child.add_argument("--out-dir", default="docs/off2-learn")
    args = parser.parse_args(argv)
    if args.command == "cv":
        return _cv(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
