# Paper 64 Terminal Audit

Date: 2026-06-15

Paper: `64_cross_embodiment_force_vocabularies`

Decision: `STRONG_REVISE`

ICLR-main ready: no

## Commands Executed

- `python -m py_compile src\run_experiment.py`
- CSV finite/schema audit over `results/force_vocab_raw.csv`, `results/force_vocab_metrics.csv`, `results/force_vocab_pairwise.csv`, `results/force_vocab_ablation.csv`, `results/force_vocab_seed_metrics.csv`, `results/force_vocabulary_training.csv`, `results/force_vocabulary_tokens.csv`, `results/negative_cases.csv`, and compatibility CSVs.
- `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `paper`
- `Copy-Item paper\main.pdf C:\Users\wangz\Downloads\64.pdf -Force`

## Verified Evidence

- Real MuJoCo contact-dynamics benchmark is implemented in `src/run_experiment.py`.
- Main evidence contains 2,940 paired rows: 7 stress/held-out embodiment splits, 5 seeds, 12 episodes per seed/split/method, and 7 methods.
- Ablation evidence contains 420 rows on the combined-shift split.
- Vocabulary evidence contains 1,200 fitting rows and 35 token-statistic rows.
- Baselines include random shooting, geometry MPC, source-action transfer, raw-force scalar transfer, robust domain-randomized MPC, and oracle embodiment MPC.
- The stale hostile-review response was updated to reflect the current v4/continuation state.
- The rebuilt PDF is `C:/Users/wangz/Downloads/64.pdf`.
- `C:/Users/wangz/Desktop/64.pdf` is absent.

## Blocking Results

CEFV remains a partial positive result, not an ICLR-main-ready result:

- Combined shift: CEFV `0.267 +/- 0.113` success and `0.168 +/- 0.018` energy; robust MPC `0.267 +/- 0.113` success and `0.163 +/- 0.015` energy.
- Heavy object: CEFV `0.233 +/- 0.108` success and `0.176 +/- 0.015` energy; robust MPC `0.300 +/- 0.117` success and `0.161 +/- 0.012` energy.
- Held-out large soft: CEFV `0.467 +/- 0.127` success and `0.125 +/- 0.008` energy; robust MPC `0.517 +/- 0.128` success and `0.123 +/- 0.008` energy.
- Held-out high gain: both CEFV and robust MPC have zero success, but robust has lower energy.
- CEFV strongly improves over raw-force scalar and geometry/source baselines, but the ICLR-main claim needs superiority over the strongest non-oracle baseline.
- Combined-shift ablations undercut the full mechanism: `no_tangent_rotation_features` and `cefv_no_online_adaptation` slightly outperform `cefv_full` on success/energy.

## Gate Decision

This paper satisfies the local evidence-package requirements for `STRONG_REVISE`: high-fidelity simulator evidence, a fitted force/effect vocabulary, held-out embodiment stress tests, strong baselines, ablations, paired tests, figures, hostile-review pressure, rebuilt PDF, and public repository.

It does not satisfy the ICLR-main-ready bar because CEFV does not consistently beat robust domain-randomized MPC, the full method is not isolated by ablations, and validation remains custom MuJoCo only.

Required revival work:

- improve CEFV until it clearly beats robust domain-randomized MPC on held-out embodiments and combined shift;
- add hardware or public benchmark validation;
- isolate vocabulary, embodiment normalization, tangent/rotation features, and online adaptation with stronger ablations;
- complete manual full-paper related-work synthesis.
