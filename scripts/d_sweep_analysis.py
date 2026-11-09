"""Role D: analyse the design sensitivity sweep and apply the predeclared rules.

Reads only saved Dev summaries (results/sweep/*/summary.json and the reused R runs in
results/stage2/d_stage1_ref_s*). Writes results/role_d/sweep_analysis.json and three
figures. Test is never read. Rules: docs/D_SWEEP_PLAN.md section 2.

Usage (from the repository root):
    python -m scripts.d_sweep_analysis
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (42, 43, 44)
OUT = Path("results/role_d")
REFERENCE = ("results/stage2", "d_stage1_ref")  # 1 layer, width 128, full recipe (R)

DEPTH = {
    1: REFERENCE,
    2: ("results/sweep", "d_sweep_layers2"),
    3: ("results/sweep", "d_sweep_layers3"),
}
WIDTH = {
    64: ("results/sweep", "d_sweep_hidden64"),
    128: REFERENCE,
    256: ("results/sweep", "d_sweep_hidden256"),
}
ABLATION = {
    "full recipe": REFERENCE,
    "no character CNN": ("results/sweep", "d_abl_nochar"),
    "no word dropout": ("results/sweep", "d_abl_nowd"),
    "neither": ("results/sweep", "d_abl_neither"),
}


def load(folder, name, seed):
    path = Path(folder) / f"{name}_s{seed}" / "summary.json"
    with open(path, encoding="utf-8") as stream:
        s = json.load(stream)
    if s.get("test_evaluated") or not s.get("reload_verified"):
        raise ValueError(f"{path}: Test evaluated or reload not verified")
    return {
        "selection_score": s["selection_score"],
        "intent_accuracy_pct": s["dev_metrics"]["intent_accuracy_pct"],
        "slot_micro_f1_pct": s["dev_metrics"]["slot_micro_f1_pct"],
        "best_epoch": s["best_epoch"],
        "epochs_run": s["epochs_run"],
        "parameters": s["parameters"]["total"],
        "minutes": s["elapsed_seconds"] / 60,
    }


def describe(folder, name):
    runs = {seed: load(folder, name, seed) for seed in SEEDS}
    stats = {}
    for key in (
        "selection_score",
        "intent_accuracy_pct",
        "slot_micro_f1_pct",
        "minutes",
    ):
        values = [r[key] for r in runs.values()]
        stats[key] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
        }
    return {
        "run": name,
        "parameters": runs[SEEDS[0]]["parameters"],
        "seeds": runs,
        **stats,
    }


def choose(sweep, current):
    """Rule 2: best mean; current supported unless beaten by more than the larger SD;
    among values within that margin of the best, prefer the smallest model."""
    mean = {k: v["selection_score"]["mean"] for k, v in sweep.items()}
    sd = {k: v["selection_score"]["sample_sd"] for k, v in sweep.items()}
    best = max(mean, key=mean.get)
    beaten_by = [k for k in sweep if mean[k] - mean[current] > max(sd[k], sd[current])]
    near_best = [k for k in sweep if mean[best] - mean[k] <= max(sd[k], sd[best])]
    preferred = min(near_best, key=lambda k: sweep[k]["parameters"])
    return {
        "best_mean": best,
        "current": current,
        "current_supported": not beaten_by,
        "values_beating_current_beyond_sd": beaten_by,
        "within_sd_of_best": sorted(near_best),
        "preferred": preferred,
    }


def verdict(full, reduced):
    """Rule 3: compare full recipe with the reduced one, seed by seed."""
    diffs = {
        seed: full["seeds"][seed]["selection_score"]
        - reduced["seeds"][seed]["selection_score"]
        for seed in SEEDS
    }
    mean = statistics.mean(diffs.values())
    if all(d > 0 for d in diffs.values()):
        label = "consistent gain"
    elif mean > 0:
        label = "average gain only"
    else:
        label = "no gain"
    return {"per_seed_pp": diffs, "mean_pp": mean, "verdict": label}


def sweep_plot(sweep, choice, xlabel, title, path, pref_name, log2=False):
    xs = sorted(sweep)
    means = [sweep[x]["selection_score"]["mean"] for x in xs]
    sds = [sweep[x]["selection_score"]["sample_sd"] for x in xs]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for x in xs:
        seeds = [sweep[x]["seeds"][s]["selection_score"] for s in SEEDS]
        ax.scatter([x] * len(seeds), seeds, color="0.65", s=18, zorder=2)
    ax.errorbar(
        xs,
        means,
        yerr=sds,
        color="C0",
        marker="o",
        capsize=4,
        zorder=3,
        label="3-seed mean ± SD",
    )
    pref = choice["preferred"]
    ax.scatter(
        [pref],
        [sweep[pref]["selection_score"]["mean"]],
        s=160,
        facecolors="none",
        edgecolors="C3",
        linewidths=2,
        zorder=4,
        label=f"preferred = {pref}",
    )
    if log2:
        ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Dev selection score (%)\nmean of intent accuracy and slot F1")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(
        0.5,
        0.005,
        f"Preferred {pref_name} = {pref} (grey dots: seeds 42/43/44; Dev only)",
        ha="center",
        fontsize=9,
        color="C3",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ablation_plot(ablation, path):
    names = list(ablation)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, seed in enumerate(SEEDS):
        ys = [ablation[n]["seeds"][seed]["selection_score"] for n in names]
        ax.plot(
            range(len(names)),
            ys,
            marker="o",
            color=f"C{i}",
            alpha=0.8,
            label=f"seed {seed}",
        )
    means = [ablation[n]["selection_score"]["mean"] for n in names]
    ax.plot(
        range(len(names)), means, marker="s", color="black", linewidth=2, label="mean"
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Dev selection score (%)")
    ax.set_title("Character CNN and word dropout, 3 seeds (Dev)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    depth = {k: describe(*v) for k, v in DEPTH.items()}
    width = {k: describe(*v) for k, v in WIDTH.items()}
    ablation = {k: describe(*v) for k, v in ABLATION.items()}
    result = {
        "note": "Dev only; Test never read. Rules in docs/D_SWEEP_PLAN.md. "
        "Frozen Stage 1 and Stage 2 systems are not changed.",
        "depth": {"by_layers": depth, "rule2": choose(depth, 1)},
        "width": {"by_hidden_dim": width, "rule2": choose(width, 128)},
        "ablation": {
            "by_config": ablation,
            "character_cnn": verdict(
                ablation["full recipe"], ablation["no character CNN"]
            ),
            "word_dropout": verdict(
                ablation["full recipe"], ablation["no word dropout"]
            ),
            "both_vs_neither": verdict(ablation["full recipe"], ablation["neither"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sweep_analysis.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    sweep_plot(
        depth,
        result["depth"]["rule2"],
        "Number of BiLSTM layers (width 128)",
        "Dev score vs BiLSTM depth",
        OUT / "sweep_depth.png",
        "number of BiLSTM layers",
    )
    sweep_plot(
        width,
        result["width"]["rule2"],
        "Hidden size per direction (1 layer)",
        "Dev score vs BiLSTM width",
        OUT / "sweep_width.png",
        "hidden size",
        log2=True,
    )
    ablation_plot(ablation, OUT / "sweep_ablation.png")

    for label, sweep, key in (("depth", depth, "layers"), ("width", width, "hidden")):
        print(f"== {label}")
        for k, v in sorted(sweep.items()):
            s = v["selection_score"]
            print(
                f"  {key}={k:>3}  params={v['parameters']:>9,}  score={s['mean']:.2f} "
                f"(SD {s['sample_sd']:.2f})  intent={v['intent_accuracy_pct']['mean']:.2f}  "
                f"slot={v['slot_micro_f1_pct']['mean']:.2f}  min={v['minutes']['mean']:.0f}"
            )
        print("  rule2:", result[label]["rule2"])
    print("== ablation")
    for k, v in ablation.items():
        s = v["selection_score"]
        per = ", ".join(f"{v['seeds'][x]['selection_score']:.2f}" for x in SEEDS)
        print(
            f"  {k:<17} params={v['parameters']:>9,}  score={s['mean']:.2f} "
            f"(SD {s['sample_sd']:.2f})  seeds: {per}"
        )
    for k in ("character_cnn", "word_dropout", "both_vs_neither"):
        print(f"  {k}: {result['ablation'][k]}")


if __name__ == "__main__":
    main()
