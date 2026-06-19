from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOWNLOADS = Path.home() / "Downloads"
DESKTOP = Path.home() / "Desktop"
PDF = DOWNLOADS / "64.pdf"


EXPECTED_ROWS = {
    "force_vocab_raw.csv": 17600,
    "force_vocab_ablation_raw.csv": 3520,
    "force_vocab_metrics.csv": 110,
    "force_vocab_seed_metrics.csv": 880,
    "force_vocab_ablation.csv": 22,
    "force_vocab_pairwise.csv": 100,
    "decision_audit.csv": 12,
    "force_vocabulary_training.csv": 10800,
    "force_vocabulary_tokens.csv": 44,
}

REQUIRED_FIGURES = [
    "force_vocab_success_by_split.png",
    "force_vocab_energy_by_split.png",
    "force_vocab_ablation_energy.png",
    "force_vocab_learning_curve.png",
    "force_vocab_token_energy.png",
]


def fail(message: str) -> None:
    print(f"VALIDATION FAILED: {message}")
    sys.exit(1)


def count_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    matches = re.findall(rb"/Type\s*/Page\b", data)
    return len(matches)


def main() -> None:
    for rel, expected in EXPECTED_ROWS.items():
        path = RESULTS / rel
        if not path.exists():
            fail(f"missing {path}")
        observed = count_csv(path)
        if observed != expected:
            fail(f"{rel} row count {observed}, expected {expected}")

    for rel in REQUIRED_FIGURES:
        path = FIGURES / rel
        if not path.exists() or path.stat().st_size == 0:
            fail(f"missing or empty figure {path}")

    summary = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    if "terminal_decision=STRONG_REVISE" not in summary:
        fail("summary.txt does not contain terminal_decision=STRONG_REVISE")

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "citebordercolor={1 0.10 0.00}" not in tex or "pdfborder={0 0 1.6}" not in tex:
        fail("bright boxed citation hyperref settings are missing")
    if r"\input{generated_tables.tex}" not in tex:
        fail("main.tex does not input generated tables")
    if "STRONG REVISE" not in tex and "STRONG\\_REVISE" not in tex:
        fail("main.tex does not state the terminal decision")

    log = PAPER / "main.log"
    if log.exists():
        log_text = log.read_text(encoding="utf-8", errors="ignore")
        fatal_patterns = ["Undefined control sequence", "LaTeX Error", "Citation `", "undefined references"]
        hits = [pat for pat in fatal_patterns if pat in log_text]
        if hits:
            fail(f"LaTeX log contains fatal patterns: {hits}")

    if not PDF.exists():
        fail(f"missing Downloads PDF {PDF}")
    pages = pdf_pages(PDF)
    if pages < 25:
        fail(f"PDF has {pages} pages, expected at least 25")

    desktop_pdf = DESKTOP / "64.pdf"
    if desktop_pdf.exists():
        fail(f"Desktop PDF exists and must not: {desktop_pdf}")

    sha = hashlib.sha256(PDF.read_bytes()).hexdigest().upper()
    print("VALIDATION PASSED")
    print(f"pdf={PDF}")
    print(f"pages={pages}")
    print(f"sha256={sha}")


if __name__ == "__main__":
    main()
