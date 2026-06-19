# Paper 64 Frozen Protocol

Date frozen: 2026-06-20

This document freezes the final evidence protocol for Paper 64. Results produced after this freeze are confirmatory evidence, not development feedback for further method tuning.

## Method Under Test

Primary method: `rc_fev_v5`

Name in manuscript: Robust Calibrated Force-Effect Vocabulary (RC-FEV)

The method combines:

- Cross-embodiment normalized force/effect tokens.
- CVaR source-branch energy.
- Mean source-branch energy.
- Geometric push feasibility.
- Token uncertainty.
- Robust worst-branch anchor.
- Online residual calibration by token and coarse action family.

## Final Command

```powershell
python src\run_experiment.py --train-tasks 240 --vocab-size 10 --seeds 8 --episodes 20 --splits nominal heldout_small_radius heldout_large_soft heldout_high_gain heldout_weak_actuator low_friction heavy_object high_friction actuation_noise combined_shift --ablation-splits combined_shift heavy_object --results-dir results --figures-dir figures
```

## Frozen Scale

- Training tasks: 240.
- Candidate actions per task: 45.
- Vocabulary size: 10.
- Seeds: 8.
- Episodes per seed per split: 20.
- Main splits: 10.
- Main methods: 11.
- Expected main policy rows: 17,600.
- Ablation splits: 2.
- Ablation methods: 11.
- Expected ablation policy rows: 3,520.
- Expected source rollout branches per candidate: 5.
- CPU-only execution; no GPU requirement.

## Main Splits

- `nominal`
- `heldout_small_radius`
- `heldout_large_soft`
- `heldout_high_gain`
- `heldout_weak_actuator`
- `low_friction`
- `heavy_object`
- `high_friction`
- `actuation_noise`
- `combined_shift`

## Main Methods

- `random_shooting`
- `geometry_mpc`
- `source_action_transfer`
- `raw_force_scalar`
- `continuous_force_regression`
- `robust_domain_randomized_mpc`
- `cvar_domain_randomized_mpc`
- `cefv_v4`
- `rc_fev_v5`
- `rc_fev_no_online`
- `oracle_embodiment_mpc`

Oracle embodiment MPC is an upper bound and must not be described as a fair deployable baseline.

## Ablation Methods

Ablations are frozen on `combined_shift` and `heavy_object`.

- `rc_fev_v5`
- `rc_fev_no_online`
- `rc_fev_no_robust_anchor`
- `rc_fev_no_embodiment_normalization`
- `rc_fev_no_tangent_rotation_features`
- `action_only_vocabulary`
- `small_vocabulary_k3`
- `cefv_v4`
- `robust_domain_randomized_mpc`
- `cvar_domain_randomized_mpc`
- `oracle_embodiment_mpc`

## Metrics

Primary:

- Success rate.
- Energy.
- Energy regret relative to oracle embodiment MPC.
- Final distance.
- Failure rate.

Secondary:

- Normalized progress.
- Contact steps.
- Normal impulse.
- Tangent impulse.
- Yaw magnitude.
- Token uncertainty.
- Robust-anchor weight.
- Pairwise success and energy deltas against each baseline.
- Bootstrap confidence intervals.
- Sign-flip p-values.

## Decision Gates

Weak baseline gate:

- RC-FEV v5 must beat or tie weak transfer baselines in aggregate energy and success:
- `geometry_mpc`
- `source_action_transfer`
- `raw_force_scalar`
- `continuous_force_regression`
- `cefv_v4`

Strong-review gate:

- RC-FEV v5 should beat or tie both robust baselines in aggregate success and energy:
- `robust_domain_randomized_mpc`
- `cvar_domain_randomized_mpc`

Ablation gate:

- RC-FEV v5 should not be consistently dominated by its own ablations on `combined_shift` and `heavy_object`.

Terminal decision:

- `ACCEPTABLE_SUBMISSION_CANDIDATE`: weak, strong, and ablation gates pass and the manuscript has no validation failures.
- `STRONG_REVISE`: weak gates pass, but strong-review or ablation gates expose limitations.
- `KILL_ARCHIVE`: weak gates fail, validation fails irrecoverably, or the final artifact cannot be reproduced.

## Reporting Rules

- Do not cherry-pick favorable splits.
- Report all frozen splits.
- Report robust/CVaR MPC even when they beat RC-FEV.
- Report online calibration effect even if modest.
- Report ablations that match or beat the full method.
- Preserve negative cases in `negative_cases.csv`.
- The manuscript must use bright boxed clickable citations and route in-text citations to the references.
- Final numbered PDF must be `C:\Users\wangz\Downloads\64.pdf`.
- No PDF may be copied to the visible Desktop.

