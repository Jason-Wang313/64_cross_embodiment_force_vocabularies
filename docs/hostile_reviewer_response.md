# Hostile Reviewer Response

Paper: 64 Cross-Embodiment Force Vocabularies

Continuation audit date: 2026-06-15

## Strongest Technical Threats

- Cross-embodiment world models and policy-transfer systems.
- Domain-randomized MPC and robust model-based control.
- Force-control and tactile/force-conditioned manipulation systems.
- Large-scale robot augmentation/data systems such as OXE-style cross-embodiment collections.
- Failure-reasoning and deployment-shift datasets for manipulation.

## ICLR Main Response

A hostile ICLR reviewer would no longer be correct to reject the paper for synthetic-only evidence. The v4 rebuild contains a real MuJoCo contact-dynamics benchmark, a fitted force/effect vocabulary, held-out embodiment splits, implemented baselines, stress tests, ablations, paired tests, figures, and a rebuilt PDF.

The reviewer would still be correct to reject the paper as ICLR-main-ready. CEFV improves over weak geometry/source/raw-force baselines, but it does not consistently beat robust domain-randomized MPC. The combined-shift ablations also fail to prove that tangent/rotation features or online adaptation are necessary, because simpler variants match or slightly outperform the full method.

## Honest Action

The current terminal state is `STRONG_REVISE`, not `KILL_ARCHIVE` and not submission-ready. The evidence is real enough to keep as a serious empirical scaffold, but not decisive enough for ICLR main.

## What Would Be Needed To Revive

- Clear wins over robust domain-randomized MPC on held-out embodiments and combined shift.
- Ablations that isolate the discrete vocabulary, embodiment normalization, tangent/rotation features, and online adaptation.
- Hardware or public benchmark validation.
- Manual full-paper related-work synthesis.
