# Paper 64 Final Submission Audit

Date: 2026-06-20

## Terminal Decision

STRONG_REVISE

Paper 64 is not ICLR-main ready. It is a rigorous, reproducible strong-revise artifact with real evidence and clear limitations.

## Final Evidence

- Main policy rows: 17,600
- Ablation policy rows: 3,520
- Split/method metric rows: 110
- Seed metric rows: 880
- Pairwise rows: 100
- Training rows: 10,800
- Token summary rows: 44
- Main PDF pages: 25
- Numbered PDF: `C:\Users\wangz\Downloads\64.pdf`
- PDF SHA256: `C10FCCB19974D2B12E97547840B4F3A0C521868C92B9716B07F2759398941F4C`
- Desktop PDF: absent

## Summary Result

RC-FEV v5 improves over weak baselines and CEFV v4 in aggregate. It slightly improves over robust and CVaR MPC in aggregate, but the margin is small. The ablation gate fails because no-online, no-normalization, no-tangent/rotation, no-robust-anchor, and small-vocabulary variants match or slightly beat the full method on hostile splits.

## Validation

Command:

```powershell
python scripts\validate_submission_artifacts.py
```

Result:

```text
VALIDATION PASSED
pdf=C:\Users\wangz\Downloads\64.pdf
pages=25
sha256=C10FCCB19974D2B12E97547840B4F3A0C521868C92B9716B07F2759398941F4C
```

## Public Repository

`https://github.com/Jason-Wang313/64_cross_embodiment_force_vocabularies`

