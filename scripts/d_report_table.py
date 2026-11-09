"""Role D: the assignment's Stage 1 / Stage 2 comparison table and the Dev seed figure.

Every value is read from saved, genuine result files; nothing is typed by hand.
Writes:
    results/role_d/report_table.csv     (assignment sheet format)
    results/role_d/report_table.md
    results/role_d/stage2_seeds.png     (Dev selection score per seed: R, C1, C2)

Usage (from the repository root):
    python -m scripts.d_report_table
"""

import csv
import json
import sys
from pathlib import Path

STAGE1_FINAL = Path("results/final_weight2/final_results.json")
STAGE2_FINAL = Path("results/stage2_c_final/notebook/final_results.json")
ANALYSIS = Path("results/role_d/c_analysis.json")
OUT = Path("results/role_d")
COLUMNS = [
    "Model version",
    "Split",
    "Total # parameters",
    "Total # trainable parameters",
    "Intent accuracy (%)",
    "Slot micro-F1 (%)",
    "Number of epochs",
]
LABELS = {
    "R": "Stage 1 recipe (same machine)",
    "C1": "C1 class-weighted intent loss",
    "C2": "C2 = C1 + 100 epochs",
}


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def stage_rows(label, final, epochs_run, best_epoch):
    rows = []
    for split, name in (("dev", "Validation"), ("test", "Test")):
        metrics = final[split]
        if metrics["split"] != split:
            raise ValueError(f"{label}: {split} record reports split {metrics['split']}")
        if split == "test" and not metrics["frozen_for_test"]:
            raise ValueError(f"{label}: Test was not evaluated on a frozen checkpoint")
        if metrics["best_epoch"] != best_epoch:
            raise ValueError(f"{label}: best epoch mismatch")
        rows.append(
            {
                "Model version": label,
                "Split": name,
                "Total # parameters": metrics["parameters"]["total"],
                "Total # trainable parameters": metrics["parameters"]["trainable"],
                "Intent accuracy (%)": f"{metrics['intent_accuracy_pct']:.2f}",
                "Slot micro-F1 (%)": f"{metrics['slot_micro_f1_pct']:.2f}",
                "Number of epochs": f"{epochs_run} (best epoch {best_epoch})",
            }
        )
    return rows


def table():
    missing = [str(p) for p in (STAGE1_FINAL, STAGE2_FINAL) if not p.exists()]
    if missing:
        print("MISSING:", ", ".join(missing))
        sys.exit(1)
    stage1 = read_json(STAGE1_FINAL)
    stage2 = read_json(STAGE2_FINAL)
    for name, final in (("Stage 1", stage1), ("Stage 2", stage2)):
        if final["freeze"]["test_used_for_selection"]:
            raise ValueError(f"{name} freeze reports Test used for selection")
    stage1_summary = read_json(Path(stage1["freeze"]["selected_run"]) / "summary.json")
    rows = stage_rows(
        "Stage-1 system", stage1, stage1_summary["epochs_run"], stage1_summary["best_epoch"]
    ) + stage_rows(
        "Stage-2 system", stage2, stage2["summary"]["epochs_run"], stage2["summary"]["best_epoch"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "report_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(COLUMNS) + " |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(f"{r[c]:,}" if isinstance(r[c], int) else str(r[c]) for c in COLUMNS)
            + " |"
        )
    deltas = "; ".join(
        f"{name} intent {stage2[s]['intent_accuracy_pct'] - stage1[s]['intent_accuracy_pct']:+.2f}, "
        f"slot micro-F1 {stage2[s]['slot_micro_f1_pct'] - stage1[s]['slot_micro_f1_pct']:+.2f}"
        for s, name in (("dev", "Validation"), ("test", "Test"))
    )
    markdown = (
        "\n".join(lines)
        + "\n\nStage 2 minus Stage 1 (percentage points, from unrounded values): "
        + deltas
        + ".\n"
    )
    (OUT / "report_table.md").write_bytes(markdown.encode("utf-8"))
    print(markdown)


def figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = read_json(ANALYSIS)
    labels = [k for k in ("R", "C1", "C2") if analysis["runs"].get(k)]
    chosen = analysis["rules"].get("stage2_configuration")
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.75 * len(labels)), layout="constrained")
    for y, label in enumerate(labels):
        runs = analysis["runs"][label]
        seeds = sorted(runs, key=int)
        values = [runs[s]["selection_score"] for s in seeds]
        ax.scatter(values, [y] * len(values), color="#555555", s=22, zorder=3)
        for seed, value in zip(seeds, values):
            ax.annotate(
                seed,
                (value, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color="#555555",
            )
        if label in analysis["three_seed_means"]:
            mean = analysis["three_seed_means"][label]["selection_score"]["mean"]
            ax.scatter([mean], [y], marker="|", s=220, color="#1f4e79", zorder=4)
            ax.annotate(
                f"mean {mean:.2f}",
                (mean, y),
                textcoords="offset points",
                xytext=(0, -16),
                ha="center",
                fontsize=8,
                color="#1f4e79",
            )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(
        [LABELS[k] + ("  (Stage 2)" if k == chosen else "") for k in labels], fontsize=8
    )
    ax.set_ylim(-0.7, len(labels) - 0.4)
    ax.set_xlabel("Dev selection score (%), mean of intent accuracy and slot micro-F1")
    ax.set_title("Role C's proposals on Dev, by seed (same machine)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUT / "stage2_seeds.png", dpi=180)
    plt.close(fig)
    print("Wrote", OUT / "stage2_seeds.png")


if __name__ == "__main__":
    figure()
    table()
