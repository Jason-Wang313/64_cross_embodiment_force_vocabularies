# Experiment Rigor Checklist

## v2 Synthetic Rigor
- [x] Multiple seeds.
- [x] Error bars.
- [x] Stronger synthetic baselines.
- [x] Ablations.
- [x] Stress tests.
- [x] Negative cases.

## ICLR Main Bar
- [ ] Real-robot validation.
- [x] High-fidelity simulator benchmark.
- [x] Implemented fitted force/effect vocabulary.
- [x] Implemented real competing baselines.
- [ ] Manual exhaustive related-work synthesis.
- [x] Paper-specific empirical figures.

Decision: real-evidence STRONG_REVISE. The main blocker is not synthetic evidence anymore; it is that the mechanism does not consistently beat robust domain-randomized MPC and lacks hardware/public-benchmark validation.
