# Paper 64 Expanded Submission Plan

Date: 2026-06-20

Paper: `64_cross_embodiment_force_vocabularies`

Target venue posture: ICLR main-conference candidate only if the rebuilt method clears robust domain-randomized MPC, CVaR-style MPC, simple source-transfer baselines, old CEFV, and ablation gates under a frozen hostile-review protocol. If it does not, produce a 25+ page strong-revise or kill/archive paper with real evidence and no rhetorical rescue.

Operating constraints:

- CPU only.
- Keep RAM light with compact MuJoCo rollouts, CSV streaming, small k-means/ridge models, and no GPU/deep model dependency.
- Do not reduce experimental quality because of RAM limits. Prefer stronger baselines, more splits, more seeds, and more ablations over large neural models.
- Do not pad the paper. The 25+ pages must be theory, protocol, real evidence, generated tables, failure analysis, related work, or reproducibility detail.
- Keep `C:/Users/wangz/Downloads/64.pdf` as the only numbered PDF artifact. Never copy to Desktop.
- Freeze the final protocol before the terminal run.

## 1. Starting Diagnosis

Verified current v4 facts:

- Repository is clean before v5 work.
- Current PDF is only 4 pages.
- Current main evidence has 2,940 rows: 7 splits, 5 seeds, 12 episodes, 7 main methods.
- Current ablation evidence has 7 aggregate rows, not raw ablation rows.
- Current terminal decision is `STRONG_REVISE`.
- CEFV beats geometry, source-action, and raw-force baselines.
- CEFV does not consistently beat robust domain-randomized MPC.
- On combined shift, CEFV ties robust success but has worse energy.
- On heavy object, robust beats CEFV on success and energy.
- The `no_tangent_rotation_features` and `cefv_no_online_adaptation` ablations slightly beat full CEFV on combined-shift energy, so the full mechanism is not isolated.
- Current manuscript is far below submission depth and uses placeholder-ish prior-work entries.

Core failure:

The v4 force/effect vocabulary is an interpretable representation but not a strong decision rule. It uses token means plus source mean/geometry terms, but robust domain-randomized MPC already captures much of the action-selection signal. Online token residual adaptation and tangent/rotation features are not proven necessary.

## 2. Revised Thesis

The original thesis should be narrowed.

Revised testable thesis:

> Cross-embodiment contact transfer benefits from force/effect tokens only when the vocabulary is calibrated as a risk-aware action scorer: tokens must predict not merely average source energy but uncertainty, robust branch disagreement, and deployment-specific residuals. Uncalibrated force vocabularies are often matched by robust MPC.

Honest outcomes:

- `ICLR_MAIN_TARGET_READY` is possible only with a large frozen positive result plus robust theory, but remains unlikely without hardware or public benchmark evidence.
- `STRONG_REVISE` if the v5 calibrated vocabulary improves over robust/CVaR MPC on aggregate success or energy without worse failure/safety and ablations isolate the calibrated token mechanism.
- `KILL_ARCHIVE` if robust/CVaR/simple baselines still match or beat the method, or if ablations show the vocabulary components are unnecessary.

## 3. Method Rebuild: RC-FEV v5

Implement a new proposed method: Robust Calibrated Force-Effect Vocabulary (RC-FEV).

Required components:

- Expanded action library:
  - add more target-relative angles;
  - add lateral offsets;
  - add multiple push-distance scales;
  - keep the same candidate set for every method.

- Force/effect tokens:
  - retain CEFV v4 tokenization as an explicit baseline;
  - train v5 tokens on normalized normal/tangential impulse, contact duration, peak forces, progress, yaw, action offset, and action distance;
  - store token mean energy, token success, token variance/uncertainty, and token counts.

- Calibrated scorer:
  - combine token energy, token uncertainty, branch disagreement, source mean, robust worst-case score, geometric score, and online deployment residual;
  - include an explicit robust anchor instead of pretending robust MPC is irrelevant;
  - report whether the vocabulary improves over the robust anchor rather than hiding that dependency.

- Online adaptation:
  - maintain token-level residuals for executed actions;
  - maintain an embodiment-family residual or branch-disagreement residual if implementation remains clean;
  - compare full online adaptation with no-online adaptation.

- Risk statistics:
  - add CVaR branch score baseline;
  - add branch-disagreement diagnostics;
  - report calibration and token-coverage metrics.

Fairness constraints:

- Robust domain-randomized MPC and CVaR MPC must remain strong baselines.
- The old CEFV v4 method must remain as a baseline.
- Continuous force regression and raw-force scalar baselines must remain.
- All methods must choose from identical candidates on identical tasks.

## 4. Main Method Suite

Frozen main methods should include:

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

Optional if clean:

- `adaptive_residual_mpc`
- `rc_fev_no_robust_anchor`

## 5. Stress Splits

Main frozen splits:

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

The v4 benchmark already contains several of these; v5 should add held-out weak actuator, high-friction, and actuation-noise stress if runtime remains feasible.

## 6. Ablations

Run ablations on at least combined shift and heavy object, because these are the v4 failure modes.

Required ablations:

- full RC-FEV;
- no online adaptation;
- no robust anchor;
- no embodiment normalization;
- no tangent/rotation/yaw/contact features;
- action-only vocabulary;
- small vocabulary;
- old CEFV v4;
- robust MPC;
- CVaR MPC;
- oracle.

Gate: the full method must beat or tie the best non-oracle ablation on energy and success. If no-online or no-tangent wins again, the full mechanism is not isolated.

## 7. Statistical Protocol

Development phase:

- Run crash-only smoke tests with tiny seeds/episodes and dev output directories.
- Run a medium pre-freeze development run on nominal, heavy object, heldout large soft, and combined shift.
- If the method fails because of an identifiable implementation weakness, improve it before freeze and log the change.
- Write all pre-freeze choices in `docs/paper64_development_log.md`.

Final freeze:

- Write `docs/paper64_protocol_freeze_20260620.md`.
- Freeze seeds, episodes, splits, methods, candidates, vocabulary size, train tasks, baselines, ablations, and decision gates.
- After freeze, only fix recoverable infrastructure failures.

Final evidence target:

- At least 8 seeds.
- At least 20 episodes per seed/split/method.
- At least 10 main splits.
- At least 11 main methods.
- Expected main raw rows: 8 x 20 x 10 x 11 = 17,600.
- Expected ablation raw rows: at least 8 x 20 x 2 x 11 = 3,520.

Statistics:

- Success/final-distance/failure/energy means and 95 percent CIs.
- Paired bootstrap intervals.
- Sign-flip p-values.
- Per-seed summaries.
- Token coverage and calibration summaries.
- Split-by-split paired deltas against robust and CVaR MPC.

## 8. Theory Additions

Required theory sections:

- Formal cross-embodiment contact-action transfer setup.
- Force/effect vocabulary as a representation of action-conditioned contact outcomes.
- Embodiment normalization assumptions and when they fail.
- Token-risk decomposition separating mean effect, branch disagreement, token uncertainty, and action-library discretization.
- Proposition: if token prediction error is lower than robust branch conservatism by a margin, token scoring can improve action selection.
- Negative theorem: if target embodiment contact modes lie outside source token support, any fixed token vocabulary can be worse than robust MPC.
- Online residual adaptation bound and stale/residual overfitting failure mode.

## 9. Related Work and Citations

Replace the current placeholder bibliography with stable primary sources on:

- MuJoCo/contact simulation;
- model predictive control and robust MPC;
- domain randomization and sim-to-real;
- force control and impedance control;
- contact-rich manipulation;
- cross-embodiment transfer;
- skill/action abstractions and options;
- vector quantization/k-means or discrete representation learning.

Citation UX requirement:

- Use `hyperref` with `colorlinks=false`.
- Use bright boxed citation borders, e.g. `citebordercolor={1 0.48 0}`.
- Verify citations route to the bibliography and LaTeX has no undefined references/citations.

## 10. Manuscript Rebuild

Minimum manuscript contents:

- Abstract with honest terminal decision.
- Introduction explaining why v4 was only strong-revise.
- Formal problem setup.
- RC-FEV v5 method.
- Theory and negative results.
- Frozen protocol.
- Main results.
- Robust/CVaR comparison.
- Ablations.
- Token coverage and calibration.
- Failure analysis.
- Related work.
- Limitations and revival conditions.
- Reproducibility appendix.
- Full generated result tables.

Length requirement:

- At least 25 pages.
- If final evidence remains only strong-revise or kill/archive, the paper should become more rigorous and honest, not more optimistic.

## 11. Validation Gates

Before commit:

- `python -m py_compile src/run_experiment.py`
- Frozen experiment completes with expected row counts.
- Analysis/tables generated.
- 25+ page PDF builds with BibTeX and enough LaTeX passes.
- `C:/Users/wangz/Downloads/64.pdf` exists.
- `C:/Users/wangz/Desktop/64.pdf` absent.
- Downloads PDF matches repo PDF.
- LaTeX scan has no undefined citations/references or fatal errors.
- `git diff --check` passes.
- Paper repo committed and pushed to public GitHub.
- Root ledgers updated locally.

## 12. Terminal Decision Gates

`ICLR_MAIN_TARGET_READY` requires:

- clear aggregate and split-level gains over robust MPC and CVaR MPC;
- no worse failure/safety;
- ablations show robust anchor, token calibration, embodiment normalization, and online adaptation are necessary;
- token coverage analysis supports the mechanism;
- 25+ page reproducible manuscript;
- ideally external/public benchmark evidence. Without that, use `STRONG_REVISE` even for a strong local result.

`STRONG_REVISE` if:

- RC-FEV improves over weak baselines and is competitive with robust/CVaR MPC, but does not fully clear all gates or lacks external validation.

`KILL_ARCHIVE` if:

- robust/CVaR MPC or simple ablations dominate;
- old CEFV or no-feature ablations match the full method;
- target embodiment splits expose unsupported token modes;
- evidence cannot support a meaningful revise path.
