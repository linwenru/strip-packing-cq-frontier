# Replication package: the empirical cost–quality frontier of strip-packing heuristics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23252643.svg)](https://doi.org/10.5281/zenodo.23252643)

Code and data for a controlled computational study re-measuring the
two-dimensional strip-packing heuristic landscape under one protocol:
104 classical benchmark instances plus a six-instance ZDF scale tier
(n up to 10 064) and a 43-instance furniture-industry tier, a
rotation-valid shared lower bound,
serial thread-capped execution, both CPU and wall time recorded.

## Requirements

Python 3.10+; `numpy` and `scikit-learn` are required (the analysis
driver imports the companion study's learning pipeline at module level;
the learned-policy cells of the scale/industrial tiers also need the
released trained models). The reference heuristic packages themselves are
pure standard library.

## Layout

- `burke_bf/`, `bbf/`, `twbf/`, `ish/`, `fh/`, `lvl/`, `sky/` — reference
  implementations of the published heuristics (validated against the source
  papers' published values).
- `cch/` — the configurable constructive engine and the experiment
  drivers. `cch/off2_papers.py` carries the measurement subcommands
  (`timing`, `timing-rl`, `ls-curve`, `ish-curve`, `ish-dense`,
  `enum-incremental`, `zdf-tier`, `industrial-a`, `report-off4`,
  `figures-off4`, `figures-off4-extra`), and
  `cch/off2_rl.py` / `cch/off2_dqn.py` carry the learned-variant training
  pipelines referenced by the companion rows.
- `data/` — the 104 classical benchmark instances (C, BKW/N, N/T families
  from 2DPackLib), `data/zdf/` (six large instances, scale tier), and
  `data/industrial-a/` (43 furniture-industry instances, Macedo et al.
  2010, via 2DPackLib).
- `docs/off2-learn/` — all result CSVs (timing, curves, incremental
  prefix runs, profile, scale tier, industrial tier).

## Reproduce

```sh
python3 -m cch.off2_papers timing            # serial thread-capped timing
python3 -m cch.off2_papers ish-curve         # RandomLS budget curve
python3 -m cch.off2_papers ish-dense         # dense short-budget checkpoints
python3 -m cch.off2_papers enum-incremental  # shells' incremental curves
python3 -m cch.off2_papers zdf-tier          # scale tier (ZDF, n up to 10 064)
python3 -m cch.off2_papers industrial-a      # industrial tier (furniture A set)
python3 -m cch.off2_papers report-off4       # regenerate the paper's tables
python3 -m cch.off2_papers figures-off4      # regenerate the frontier figure
python3 -m cch.off2_papers figures-off4-extra  # family/scale/industrial figures
```

All reported solutions and aggregate statistics regenerate
deterministically; timing figures are medians of repeated serial runs and
are not claimed bitwise-reproducible.

## Data acknowledgement

Benchmark instances from 2DPackLib (University of Bologna,
https://site.unibo.it/operations-research/en/research/2dpacklib).

## License

MIT (see LICENSE).
