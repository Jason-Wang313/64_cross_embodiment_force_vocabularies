# ICLR Main Gate

Paper: 64 cross_embodiment_force_vocabularies

Existing v2 decision: KILL_ARCHIVE

Gate verdict: STRONG_REVISE

Evidence digest: pending-v4-real-mujoco

Resolved blockers:
- Synthetic-only evidence replaced by real MuJoCo contact dynamics.
- Implemented fitted force/effect vocabulary.
- Implemented geometry, source-transfer, raw-force, robust-MPC, and oracle comparators.
- Added multi-seed metrics, ablations, pairwise tests, and figures.

Remaining blockers:
- CEFV does not consistently beat robust domain-randomized MPC.
- Combined-shift ablations do not prove all full-method features are necessary.
- No hardware/public benchmark validation.
- Manual exhaustive related-work synthesis remains incomplete.

The honest main-conference-safe decision is strong revise rather than submission.
