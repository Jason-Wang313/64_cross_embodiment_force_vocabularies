# 64 Cross-Embodiment Force Vocabularies

Submission-hardening version: v5 robust calibrated force/effect vocabulary rebuild

Terminal decision: STRONG_REVISE for ICLR main conference.

This version replaces the short v4 draft with a frozen, CPU-only MuJoCo contact-dynamics benchmark for robust calibrated force/effect vocabularies (RC-FEV). A discrete force/effect vocabulary is fitted from source-embodiment rollouts and evaluated on held-out robot embodiments, object/friction shifts, actuation-noise shifts, and combined stress.

The result is useful but not submission-ready: RC-FEV v5 beats weak transfer baselines and CEFV v4 in aggregate and slightly beats robust/CVaR MPC in aggregate, but the ablation gate fails because simplified variants match or slightly beat the full model on hostile splits.

Final validation:

- Main rows: 17,600
- Ablation rows: 3,520
- Split/method metrics: 110
- Seed metrics: 880
- Pairwise rows: 100
- PDF: `C:/Users/wangz/Downloads/64.pdf`
- PDF pages: 25
- PDF SHA256: `C10FCCB19974D2B12E97547840B4F3A0C521868C92B9716B07F2759398941F4C`

## Reproduce Real Benchmark

```powershell
python src\run_experiment.py --train-tasks 240 --vocab-size 10 --seeds 8 --episodes 20 --splits nominal heldout_small_radius heldout_large_soft heldout_high_gain heldout_weak_actuator low_friction heavy_object high_friction actuation_noise combined_shift --ablation-splits combined_shift heavy_object --results-dir results --figures-dir figures
```

Expected full run: 240 vocabulary-fitting tasks, 8 seeds, 20 episodes per seed/split/method, 10 splits, 11 main methods, and two ablation splits.

## Rebuild PDF

```powershell
python scripts\render_latex_tables.py
powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1
python scripts\validate_submission_artifacts.py
```

Canonical local PDF: `C:/Users/wangz/Downloads/64.pdf`
