# Submission Readiness Decision

Decision: STRONG_REVISE

ICLR main-conference readiness: NO.

Reason: The paper now has real MuJoCo contact-dynamics evidence, implemented baselines, ablations, pairwise tests, and paper-specific figures. However, CEFV does not consistently beat robust domain-randomized MPC, and the combined-shift ablation without tangent/rotation features slightly outperforms the full method. The empirical story is credible but not yet an ICLR-main-ready contribution.

Honest terminal action: strong revise for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: improve the vocabulary/action-selection mechanism until it reliably beats robust MPC on held-out embodiments, add hardware or a public high-fidelity benchmark, and complete manual full-paper related-work synthesis.
