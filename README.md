# 64 Cross-Embodiment Force Vocabularies

Submission-hardening version: v4 real-evidence rebuild

Terminal decision: STRONG_REVISE for ICLR main conference.

This version replaces the synthetic stress-test scaffold with a real MuJoCo contact-dynamics benchmark. A discrete force/effect vocabulary is fitted from source-embodiment rollouts and evaluated on held-out robot embodiments, object/friction shifts, and combined stress.

The result is useful but not submission-ready: CEFV beats geometry/source/raw-force baselines on most stress splits, but it does not consistently beat robust domain-randomized MPC and its ablations do not yet prove the full mechanism is necessary.

## Reproduce Real Benchmark

```powershell
python src\run_experiment.py
```

Expected full run: 120 vocabulary-fitting tasks, 5 seeds, 12 episodes per seed/split/method, 7 splits, 7 main methods, and combined-shift ablations.

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/64.pdf`
