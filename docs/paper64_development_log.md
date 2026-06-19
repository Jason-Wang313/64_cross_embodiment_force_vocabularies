# Paper 64 Development Log

Date: 2026-06-20

This log records pre-freeze development decisions for Paper 64, "Robust Calibrated Force-Effect Vocabularies for Cross-Embodiment Manipulation." These runs are explicitly not final evidence. They were used to expose implementation errors and method weaknesses before freezing the final protocol.

## Starting Point

The v4 paper had a 4 page manuscript and a 2,940 row MuJoCo benchmark. CEFV beat geometry/source/raw-force baselines but did not consistently beat robust domain-randomized MPC. The expanded submission standard therefore required a stronger method, stronger baselines, stress tests, ablations, and an honest terminal decision.

## Planned Upgrade

The v5 method is RC-FEV: a robust calibrated force-effect vocabulary with:

- Cross-embodiment force/effect tokenization from normalized contact features.
- Robust and CVaR branch anchors so the method cannot win by weakening robust MPC.
- Online residual calibration over token and action families.
- Held-out embodiment, object-mass, friction, actuator, noise, and combined-shift tests.
- Oracle embodiment MPC only as an upper bound, not as a fair baseline.

## Pre-Freeze Runs

### Smoke Run

Command:

```powershell
python src\run_experiment.py --train-tasks 4 --vocab-size 3 --seeds 1 --episodes 1 --splits nominal --ablation-splits combined_shift --results-dir results\dev_smoke --figures-dir figures\dev_smoke
```

Outcome:

- Completed successfully.
- Produced all expected CSV and figure outputs.
- Verified basic MuJoCo rollout, scoring, aggregation, pairwise, and decision-audit paths.

### Medium Development Run

Command:

```powershell
python src\run_experiment.py --train-tasks 30 --vocab-size 6 --seeds 2 --episodes 4 --splits nominal heldout_large_soft heavy_object combined_shift --ablation-splits combined_shift heavy_object --results-dir results\dev_medium --figures-dir figures\dev_medium
```

Outcome:

- Completed successfully.
- Aggregate RC-FEV v5 success: 0.6250.
- Aggregate RC-FEV v5 energy: 0.1235.
- Robust MPC energy: 0.1243.
- CVaR MPC energy: 0.1279.
- CEFV v4 energy: 0.1337.
- Problem found: `rc_fev_v5` tied `rc_fev_no_online`, because the initial online term included a global last-error penalty that was constant across all candidates.

### Online Calibration Repair

Change:

- Added action-family residuals to `AdaptState`.
- Replaced the candidate-invariant online penalty with candidate-specific online action bias and uncertainty-aware mismatch penalty.
- Reduced robust-anchor weight modestly as online observations accumulate.

Smoke command:

```powershell
python src\run_experiment.py --train-tasks 4 --vocab-size 3 --seeds 1 --episodes 1 --splits nominal --ablation-splits combined_shift --results-dir results\dev_smoke_online --figures-dir figures\dev_smoke_online
```

Outcome:

- Completed successfully.

### Coarse Action-Family Calibration

Problem:

- The first repair used a distance-sensitive action key that almost never repeated in short horizons.

Change:

- Replaced the fine online key with coarse angle/offset/distance-family bins.

Command:

```powershell
python src\run_experiment.py --train-tasks 30 --vocab-size 6 --seeds 2 --episodes 4 --splits nominal heldout_large_soft heavy_object combined_shift --ablation-splits combined_shift heavy_object --results-dir results\dev_medium_online_coarse --figures-dir figures\dev_medium_online_coarse
```

Outcome:

- Completed successfully.
- Four-episode horizon still produced no selected-candidate changes between online and no-online variants.
- This was judged too short to evaluate online calibration.

### Online-Horizon Development Run

Command:

```powershell
python src\run_experiment.py --train-tasks 30 --vocab-size 6 --seeds 2 --episodes 20 --splits heavy_object combined_shift --ablation-splits combined_shift heavy_object --results-dir results\dev_online_horizon --figures-dir figures\dev_online_horizon
```

Outcome:

- Completed successfully.
- Candidate differences between `rc_fev_v5` and `rc_fev_no_online`: 2 of 80.
- Both changed choices improved observed energy:
- heavy_object seed 1 episode 18: online 0.112221 vs offline 0.123695.
- combined_shift seed 0 episode 19: online 0.137552 vs offline 0.140753.
- Aggregate hostile-shift RC-FEV v5 success: 0.3125.
- Aggregate hostile-shift RC-FEV v5 energy: 0.1608.
- Robust MPC energy: 0.1596.
- CVaR MPC energy: 0.1605.
- Interpretation: online adaptation is real but small; robust/CVaR MPC remain extremely competitive. The final protocol must report this rather than over-tune around it.

## Pre-Freeze Decision

Freeze RC-FEV v5 after the coarse online-action calibration repair. Do not further tune against final seeds. The final paper must state that:

- RC-FEV v5 is useful against weak transfer and v4 vocabulary baselines.
- Robust and CVaR MPC are strong and sometimes better.
- Online calibration has measurable but modest effect.
- The result should be judged as STRONG_REVISE unless the frozen full run fails weak baseline gates, in which case it becomes KILL_ARCHIVE.

