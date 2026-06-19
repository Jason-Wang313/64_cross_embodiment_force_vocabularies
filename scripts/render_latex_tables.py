from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def method_label(method: str) -> str:
    labels = {
        "random_shooting": "Random shooting",
        "geometry_mpc": "Geometry MPC",
        "source_action_transfer": "Source action",
        "raw_force_scalar": "Raw force scalar",
        "continuous_force_regression": "Continuous force reg.",
        "robust_domain_randomized_mpc": "Robust MPC",
        "cvar_domain_randomized_mpc": "CVaR MPC",
        "cefv_v4": "CEFV v4",
        "rc_fev_v5": "RC-FEV",
        "rc_fev_no_online": "RC-FEV no online",
        "oracle_embodiment_mpc": "Oracle embodiment",
        "rc_fev_no_robust_anchor": "No robust anchor",
        "rc_fev_no_embodiment_normalization": "No emb. norm.",
        "rc_fev_no_tangent_rotation_features": "No tangent/rot.",
        "action_only_vocabulary": "Action-only vocab.",
        "small_vocabulary_k3": "Small vocabulary",
    }
    return labels.get(method, method.replace("_", " "))


def split_label(split: str) -> str:
    labels = {
        "nominal": "Nominal",
        "heldout_small_radius": "Small radius",
        "heldout_large_soft": "Large soft",
        "heldout_high_gain": "High gain",
        "heldout_weak_actuator": "Weak actuator",
        "low_friction": "Low friction",
        "heavy_object": "Heavy object",
        "high_friction": "High friction",
        "actuation_noise": "Actuation noise",
        "combined_shift": "Combined shift",
    }
    return labels.get(split, split.replace("_", " "))


def escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def aggregate_table(rows: list[dict[str, str]]) -> str:
    order = [
        "random_shooting",
        "geometry_mpc",
        "source_action_transfer",
        "raw_force_scalar",
        "continuous_force_regression",
        "cefv_v4",
        "robust_domain_randomized_mpc",
        "cvar_domain_randomized_mpc",
        "rc_fev_no_online",
        "rc_fev_v5",
        "oracle_embodiment_mpc",
    ]
    by_method = {row["method"]: row for row in rows}
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Frozen aggregate results over 10 splits, 8 seeds, and 20 episodes per seed/split. Success is higher-is-better; energy, regret, final distance, and failure are lower-is-better. Oracle embodiment MPC is an upper bound, not a deployable baseline.}",
        r"\label{tab:aggregate}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Success & Energy & Regret & Final dist. & Failure \\",
        r"\midrule",
    ]
    for method in order:
        row = by_method[method]
        name = r"\textbf{RC-FEV}" if method == "rc_fev_v5" else escape(method_label(method))
        out.append(
            f"{name} & {fnum(row['success_rate'])} & {fnum(row['energy_mean'])} & "
            f"{fnum(row['energy_regret_mean'])} & {fnum(row['final_distance_mean'])} & {fnum(row['failure_rate'])} \\\\"
        )
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(out)


def split_table(rows: list[dict[str, str]]) -> str:
    selected = {"rc_fev_v5", "robust_domain_randomized_mpc", "cvar_domain_randomized_mpc", "source_action_transfer", "cefv_v4"}
    split_order = [
        "nominal",
        "heldout_small_radius",
        "heldout_large_soft",
        "heldout_high_gain",
        "heldout_weak_actuator",
        "low_friction",
        "heavy_object",
        "high_friction",
        "actuation_noise",
        "combined_shift",
    ]
    by_key = {(row["split"], row["method"]): row for row in rows}
    method_order = ["rc_fev_v5", "robust_domain_randomized_mpc", "cvar_domain_randomized_mpc", "source_action_transfer", "cefv_v4"]
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Per-split comparison against the strongest non-oracle baselines. Each cell reports success / energy.}",
        r"\label{tab:splits}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Split & RC-FEV & Robust MPC & CVaR MPC & Source action & CEFV v4 \\",
        r"\midrule",
    ]
    for split in split_order:
        cells = []
        for method in method_order:
            row = by_key[(split, method)]
            cells.append(f"{fnum(row['success_rate'])}/{fnum(row['energy_mean'])}")
        out.append(f"{escape(split_label(split))} & " + " & ".join(cells) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(out)


def ablation_table(rows: list[dict[str, str]]) -> str:
    order = [
        "rc_fev_v5",
        "rc_fev_no_online",
        "rc_fev_no_robust_anchor",
        "rc_fev_no_embodiment_normalization",
        "rc_fev_no_tangent_rotation_features",
        "action_only_vocabulary",
        "small_vocabulary_k3",
        "cefv_v4",
        "robust_domain_randomized_mpc",
        "cvar_domain_randomized_mpc",
        "oracle_embodiment_mpc",
    ]
    by_key = {(row["split"], row["method"]): row for row in rows}
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablations on the two frozen hostile splits. The ablation gate fails because several simplifications match or slightly beat the full model.}",
        r"\label{tab:ablations}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Combined succ. & Combined energy & Heavy succ. & Heavy energy \\",
        r"\midrule",
    ]
    for method in order:
        c = by_key[("combined_shift", method)]
        h = by_key[("heavy_object", method)]
        name = r"\textbf{RC-FEV}" if method == "rc_fev_v5" else escape(method_label(method))
        out.append(
            f"{name} & {fnum(c['success_rate'])} & {fnum(c['energy_mean'])} & "
            f"{fnum(h['success_rate'])} & {fnum(h['energy_mean'])} \\\\"
        )
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(out)


def decision_table(rows: list[dict[str, str]]) -> str:
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Frozen decision audit. Positive success deltas and energy improvements favor RC-FEV.}",
        r"\label{tab:decision}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Decision & Reason \\",
        r"\midrule",
    ]
    for row in rows:
        out.append(f"{escape(row['terminal_decision'])} & {escape(row['reason'])} \\\\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(out)


def main() -> None:
    aggregate = read_csv("force_vocab_aggregate.csv")
    metrics = read_csv("force_vocab_metrics.csv")
    ablation = read_csv("force_vocab_ablation.csv")
    decision = read_csv("decision_audit.csv")
    text = "\n\n".join(
        [
            "% Generated by scripts/render_latex_tables.py from frozen results CSV files.",
            aggregate_table(aggregate),
            split_table(metrics),
            ablation_table(ablation),
            decision_table(decision),
        ]
    )
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "generated_tables.tex").write_text(text, encoding="utf-8")
    print(f"wrote {PAPER / 'generated_tables.tex'}")


if __name__ == "__main__":
    main()
