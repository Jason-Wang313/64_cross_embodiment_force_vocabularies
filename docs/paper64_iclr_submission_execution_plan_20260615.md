# Paper 64 ICLR-Main Execution Plan

Date: 2026-06-15

Paper: `64_cross_embodiment_force_vocabularies`

Goal: verify whether the current real MuJoCo force-vocabulary evidence can honestly upgrade the paper from `STRONG_REVISE` to ICLR-main readiness, or reaffirm a terminal non-ready decision with exact evidence.

## Execution Gates

1. Reproducibility gate:
   - Compile `src/run_experiment.py`.
   - Confirm main, seed, paired, ablation, vocabulary-training, vocabulary-token, stress, and negative-case CSV outputs exist.
   - Confirm all CSV outputs are non-empty and finite.
   - Rebuild the PDF from `paper/main.tex` with BibTeX.

2. Evidence gate:
   - Confirm the benchmark uses real MuJoCo contact dynamics and contact-force measurements rather than synthetic tables.
   - Confirm the fitted discrete force/effect vocabulary exists.
   - Confirm held-out embodiment splits, object/friction shifts, multiple seeds, uncertainty estimates, paired comparisons, and ablations.
   - Confirm baselines include geometry MPC, source-action transfer, raw-force scalar transfer, robust domain-randomized MPC, random shooting, and oracle embodiment MPC.

3. ICLR-main claim gate:
   - Require CEFV to clearly beat robust domain-randomized MPC on held-out embodiment and combined-shift splits.
   - Require ablations to isolate embodiment normalization, tangent/rotation features, online adaptation, and the discrete vocabulary mechanism.
   - Require hostile related-work pressure and honest limitations.
   - Fix any stale docs that still describe the old synthetic `KILL_ARCHIVE` state as current.

4. Artifact gate:
   - Rebuild `paper/main.pdf`.
   - Copy only `C:/Users/wangz/Downloads/64.pdf`.
   - Confirm `C:/Users/wangz/Desktop/64.pdf` is absent.
   - Confirm the GitHub repository is public and pushed.

## Decision Rule

Upgrade only if every ICLR-main claim gate is supported by current evidence. If CEFV still ties or loses to robust domain-randomized MPC, if ablations show simpler variants matching or beating the full method, or if validation remains custom-MuJoCo-only, keep the terminal decision as `STRONG_REVISE`.
