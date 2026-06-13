# Paper 64 Rebuild Plan: Cross-Embodiment Force Vocabularies

Date: 2026-06-13

## Goal

Rebuild Paper 64 from an archived synthetic scaffold into a real ICLR-main-target empirical robotics submission, or terminate it honestly as `STRONG_REVISE` / `KILL_ARCHIVE` if the evidence does not support the claim.

Target claim:

> A learned discrete vocabulary of contact-force/effect tokens can provide an embodiment-normalized action interface that transfers contact-rich manipulation behavior across robot embodiments better than raw action transfer, generic uncertainty, and geometry-only planning.

## Starting Audit

The current repository is not submission-ready:

- `src/run_experiment.py` is a synthetic probability-table generator.
- `paper/main.tex` is an archive memo, not a full research paper.
- Existing docs explicitly mark the paper `KILL_ARCHIVE`.
- Existing rigor gaps include no high-fidelity simulator benchmark, no implemented learned model, no real competing baselines, no paper-specific qualitative figures, and no manual related-work synthesis.

This rebuild must replace the core evidence, not polish the existing archive.

## Non-Negotiable Evidence Bar

The paper may only move toward submission if all of the following artifacts are produced:

- Real MuJoCo rollouts, using contact dynamics and force measurements from the simulator.
- A learned or fitted force-vocabulary mechanism, not hard-coded success probabilities.
- Held-out embodiment transfer splits.
- Strong implemented baselines, including non-vocabulary alternatives.
- Multi-seed evaluation with confidence intervals and pairwise tests.
- Ablations that isolate the vocabulary, embodiment normalization, and force/effect features.
- Stress tests for friction, object mass, actuator gain, and morphology changes.
- Paper-specific plots generated from real result CSVs.
- A full paper draft that reports limitations honestly.

If any item is infeasible in this environment, the paper must remain `KILL_ARCHIVE`.

## Benchmark Design

Use a lightweight but real MuJoCo contact-pushing benchmark:

- Scene: planar pusher manipulates a box toward randomized target positions.
- Embodiments: end-effector radius, pusher mass, actuator gain, contact compliance, and damping vary across robot variants.
- Train embodiments: several nominal/source variants.
- Test embodiments: held-out radius/compliance/gain/mass combinations not used to fit the vocabulary.
- Stress splits: nominal, low friction, high friction, heavy object, weak actuator, morphology shift, and combined shift.
- Episode metric: normalized box-target distance improvement, success rate, contact stability, and safety/failure events.

The task should stay computationally tractable on CPU while remaining physically meaningful. Prefer many short controlled rollouts over a memory-heavy learned policy stack.

## Method To Implement

Implement `Cross-Embodiment Force Vocabulary` as a real action-selection mechanism:

1. Collect MuJoCo rollouts across source embodiments.
2. Extract force/effect descriptors from each candidate action:
   - Normal and tangential contact force summaries.
   - Contact duration and impulse.
   - Box displacement, rotation, and final distance improvement.
   - Action direction and magnitude after embodiment normalization.
3. Fit a discrete vocabulary with deterministic k-means or equivalent lightweight clustering.
4. Score candidate actions by matching desired task progress through vocabulary-token statistics.
5. Adapt token scoring online using observed contact/effect mismatch from the current embodiment.

The implementation must store raw rollout rows, vocabulary assignments, per-seed metrics, aggregate metrics, and pairwise comparison statistics.

## Baselines

Compare against real implemented competitors:

- `random_shooting`: random candidate action selection.
- `geometry_mpc`: selects actions by geometric target progress without force vocabulary.
- `source_action_transfer`: reuses source-embodiment action statistics directly.
- `raw_force_scalar`: transfers by contact-force magnitude only, without discrete force/effect tokens.
- `robust_domain_randomized_mpc`: scores candidates over multiple randomized dynamics settings.
- `oracle_embodiment_mpc`: upper bound with access to the true test embodiment during candidate scoring.
- `cefv_full`: proposed force-vocabulary method.

The claim only survives if `cefv_full` beats the non-oracle baselines on held-out embodiment and combined-shift splits with nontrivial effect size.

## Ablations

Run ablations on the strongest stress splits:

- Remove discrete vocabulary and use continuous force features directly.
- Remove embodiment normalization.
- Remove online token adaptation.
- Remove tangential/rotational effect features.
- Use action-only vocabulary without force/effect descriptors.
- Vary vocabulary size.

These ablations should answer whether the force vocabulary itself matters, not merely whether more features help.

## Statistical Plan

Use fixed seeds and report:

- At least five random seeds.
- Per-seed success rate and final distance improvement.
- 95% confidence intervals.
- Paired comparisons against each baseline on matched seed/split conditions.
- Holm-corrected or clearly labeled uncorrected p-values, depending on available tooling.
- Failure-case tables for splits where the method does not win.

The draft must not claim SOTA if effects are small, inconsistent, or only visible against weak baselines.

## Execution Stages

1. Replace the synthetic experiment runner with a MuJoCo-based runner.
2. Add deterministic dataset/vocabulary fitting and evaluation outputs.
3. Run a small smoke test to verify contacts, forces, and CSV schemas.
4. Run the full multi-seed benchmark.
5. Generate plots and statistical summaries.
6. Rewrite the paper from archive memo to evidence-bearing submission draft.
7. Compile the PDF and copy only `C:\Users\wangz\Downloads\64.pdf`.
8. Update repository docs, root batch reports, and GitHub state.

## Terminal Decision Rules

Mark `SUBMISSION_READY_CANDIDATE` only if:

- `cefv_full` is best non-oracle or statistically tied for best on most held-out embodiment splits.
- It clearly beats `geometry_mpc`, `source_action_transfer`, and `raw_force_scalar` on combined shift.
- The oracle gap is explainable and not catastrophic.
- Ablations show the vocabulary and normalization are necessary.
- The paper has honest limitations and reproducible artifacts.

Mark `STRONG_REVISE` if:

- The method is real and sometimes useful, but does not consistently beat robust baselines or has incomplete literature/manual evidence.

Mark `KILL_ARCHIVE` if:

- Observed-only, geometry-only, raw-force, or robust domain-randomized baselines match the proposed method.
- MuJoCo contact evidence is too unstable or too weak to support the mechanism.
- The method relies on oracle information or template-generated/synthetic metrics.

## Required Final Artifacts

- `src/run_experiment.py`: real MuJoCo implementation.
- `results/*.csv`: raw rollouts, vocabulary statistics, aggregate metrics, ablations, pairwise tests.
- `figures/*.png`: paper-specific plots.
- `docs/paper64_terminal_evidence.md`: final decision with evidence.
- `paper/main.tex`: rebuilt paper or honest terminal archive.
- `C:\Users\wangz\Downloads\64.pdf`: numbered PDF in Downloads only.
- Public GitHub repository updated at `https://github.com/Jason-Wang313/64_cross_embodiment_force_vocabularies`.

