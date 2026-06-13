# Paper 64 Terminal Evidence

Date: 2026-06-13

Decision: STRONG_REVISE.

ICLR main ready: no.

## What Changed

The synthetic v3 scaffold was replaced with a real MuJoCo contact-dynamics benchmark. The new runner fits a discrete force/effect vocabulary from source-embodiment rollouts, then evaluates action selection on held-out robot embodiments and contact shifts.

## Run Configuration

- Training: 120 vocabulary-fitting tasks.
- Vocabulary: 8 force/effect tokens fitted by deterministic k-means.
- Evaluation: 5 seeds, 12 episodes per seed/split/method.
- Main rows: 2,940.
- Ablation rows: 420.
- Main splits: nominal, heldout_small_radius, heldout_large_soft, heldout_high_gain, low_friction, heavy_object, combined_shift.
- Main methods: random_shooting, geometry_mpc, source_action_transfer, raw_force_scalar, robust_domain_randomized_mpc, cefv_full, oracle_embodiment_mpc.

## Key Results

- Combined shift: CEFV success 0.267 +/- 0.113, energy 0.168 +/- 0.018; robust MPC success 0.267 +/- 0.113, energy 0.163 +/- 0.015; oracle success 0.367 +/- 0.123.
- Heavy object: CEFV success 0.233 +/- 0.108, energy 0.176 +/- 0.015; robust MPC success 0.300 +/- 0.117, energy 0.161 +/- 0.012.
- Held-out large soft embodiment: CEFV success 0.467 +/- 0.127, energy 0.125 +/- 0.008; robust MPC success 0.517 +/- 0.128, energy 0.123 +/- 0.008.
- Nominal: CEFV success 0.250 +/- 0.111, energy 0.134 +/- 0.008; robust MPC success 0.250 +/- 0.111, energy 0.136 +/- 0.008.
- CEFV significantly improves energy over raw_force_scalar on every split, including combined shift (+0.107 energy improvement, p < 0.0001).
- CEFV does not consistently improve over robust_domain_randomized_mpc; robust is better on heavy_object and heldout_high_gain and essentially tied on combined_shift, heldout_large_soft, low_friction, and nominal.

## Ablation Result

Combined-shift ablations weaken the submission claim:

- no_tangent_rotation_features: success 0.283, energy 0.163.
- cefv_no_online_adaptation: success 0.283, energy 0.164.
- cefv_full: success 0.267, energy 0.168.
- action_only_vocabulary: success 0.233, energy 0.198.
- continuous_force_regression: success 0.117, energy 0.276.

The vocabulary is useful relative to continuous raw-force regression, but the full online/tangent/rotation mechanism is not proven necessary.

## Terminal Judgment

This is no longer a kill/archive due to lack of real evidence. It is a real robotics experiment with useful negative/partial-positive findings.

It is not ICLR-main-ready because:

- The proposed method does not beat the strongest non-oracle baseline.
- The ablation evidence does not isolate the full mechanism.
- The benchmark is custom MuJoCo only, with no hardware or public benchmark validation.
- Related work still needs manual full-paper synthesis.

Final status: STRONG_REVISE.
