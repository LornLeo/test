"""Role D: analyse Stage 2 runs on Role C's proposals and apply the predeclared rules.

Reads only saved Dev outputs under results/stage2/ (summary.json and the Git-ignored
dev_predictions.jsonl). Writes results/role_d/c_analysis.json. Test is never read.
Rules: report Sections 4 and 5.3.

Usage (from the repository root):
    python -m scripts.d_c_analysis
"""

import contextlib
import io
import json
import statistics
from collections import Counter
from pathlib import Path

from src.data.loader import load_massive

STAGE2 = Path("results/stage2")
OUTPUT = Path("results/role_d/c_analysis.json")
SEEDS = (42, 43, 44)
CONFIGS = {
    "R": "d_stage1_ref",  # unchanged Stage 1 recipe, same machine
    "C1": "d_intentcw",  # class-weighted intent loss
    "C2": "d_intentcw_long",  # C1 + 100 epochs, patience 12
}
WATCH = ("general_quirky", "qa_factoid")


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def finished(name, seed):
    return (STAGE2 / f"{name}_s{seed}" / "summary.json").exists()


def summary(name, seed):
    s = read_json(STAGE2 / f"{name}_s{seed}" / "summary.json")
    if s.get("test_evaluated") or not s.get("reload_verified"):
        raise ValueError(f"{name}_s{seed}: Test evaluated or reload not verified")
    return s


def per_intent(name, seed):
    with open(STAGE2 / f"{name}_s{seed}" / "dev_predictions.jsonl", encoding="utf-8") as f:
        pairs = [(r["gold"]["intent"], r["prediction"]["intent"]) for r in map(json.loads, f)]
    tp, fp, support = Counter(), Counter(), Counter()
    for gold, pred in pairs:
        support[gold] += 1
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
    accuracy = 100 * sum(tp.values()) / len(pairs)
    recorded = summary(name, seed)["dev_metrics"]["intent_accuracy_pct"]
    if abs(accuracy - recorded) > 1e-9:
        raise ValueError(f"{name}_s{seed}: saved predictions do not match the recorded accuracy")
    out = {}
    for intent in support:
        p = tp[intent] / (tp[intent] + fp[intent]) if tp[intent] + fp[intent] else 0.0
        r = tp[intent] / support[intent]
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[intent] = {"support": support[intent], "correct": tp[intent], "f1_pct": 100 * f1}
    return out


def frequency_groups(stats, train_counts):
    order = sorted(stats, key=lambda i: (train_counts[i], i))
    q = len(order) // 4
    groups = {}
    for name, group in (("rarest_quarter", order[:q]), ("most_frequent_quarter", order[-q:])):
        support = sum(stats[i]["support"] for i in group)
        correct = sum(stats[i]["correct"] for i in group)
        groups[name] = {"dev_support": support, "dev_accuracy_pct": 100 * correct / support}
    return groups


def run_record(name, seed, train_counts):
    s = summary(name, seed)
    stats = per_intent(name, seed)
    return {
        "run": f"{name}_s{seed}",
        "intent_accuracy_pct": s["dev_metrics"]["intent_accuracy_pct"],
        "slot_micro_f1_pct": s["dev_metrics"]["slot_micro_f1_pct"],
        "selection_score": s["selection_score"],
        "best_epoch": s["best_epoch"],
        "epochs_run": s["epochs_run"],
        "parameters": s["parameters"],
        "accuracy_by_train_frequency": frequency_groups(stats, train_counts),
        "watched_intent_f1_pct": {i: stats[i]["f1_pct"] for i in WATCH},
    }


def three_seed(records):
    out = {}
    for key in ("intent_accuracy_pct", "slot_micro_f1_pct", "selection_score"):
        values = [r[key] for r in records]
        out[key] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
    return out


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        train = load_massive("data/1.0/data/en-US.jsonl")["train"]
    train_counts = Counter(r["intent"] for r in train)
    runs = {
        label: {
            seed: run_record(name, seed, train_counts) for seed in SEEDS if finished(name, seed)
        }
        for label, name in CONFIGS.items()
    }
    for label in ("R", "C1"):
        if sorted(runs[label]) != list(SEEDS):
            raise FileNotFoundError(f"{label} needs seeds {SEEDS}")
    means = {label: three_seed(list(r.values())) for label, r in runs.items() if len(r) == 3}

    # Rule 1: C1 seed-42 gate against R seed 42 (predeclared in f147e58).
    gate_c1 = runs["R"][42]["selection_score"]
    rules = {
        "rule1_c1_gate": gate_c1,
        "rule1_c1_passes": runs["C1"][42]["selection_score"] > gate_c1,
    }
    # Rule 2: C2 seed-42 gate against C1 seed 42.
    if 42 in runs["C2"]:
        gate_c2 = runs["C1"][42]["selection_score"]
        rules["rule2_c2_gate"] = gate_c2
        rules["rule2_c2_passes"] = runs["C2"][42]["selection_score"] > gate_c2
    # Rules 3-4: Stage 2 = C1 unless C2 passed and has the higher 3-seed mean.
    decision = None
    if 42 in runs["C2"]:
        if not rules["rule2_c2_passes"]:
            decision = "C1"
        elif "C2" in means:
            c1_mean = means["C1"]["selection_score"]["mean"]
            c2_mean = means["C2"]["selection_score"]["mean"]
            decision = "C2" if c2_mean > c1_mean else "C1"
    rules["stage2_configuration"] = decision
    if decision:
        rules["stage2_run"] = f"{CONFIGS[decision]}_s42"
        chosen = means[decision]["selection_score"]["mean"]
        rules["rule5_mean_vs_reference_pp"] = chosen - means["R"]["selection_score"]["mean"]

    result = {
        "note": "Dev only; Test never read. Rules in docs/D_STAGE2_PLAN.md.",
        "configs": CONFIGS,
        "runs": {label: {str(k): v for k, v in r.items()} for label, r in runs.items()},
        "three_seed_means": means,
        "rules": rules,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    for label, r in runs.items():
        for seed, rec in r.items():
            groups = rec["accuracy_by_train_frequency"]
            print(
                f"{label:2s} s{seed}: intent {rec['intent_accuracy_pct']:.2f} "
                f"slot {rec['slot_micro_f1_pct']:.2f} selection {rec['selection_score']:.4f} "
                f"best {rec['best_epoch']}/{rec['epochs_run']} rare "
                f"{groups['rarest_quarter']['dev_accuracy_pct']:.2f} frequent "
                f"{groups['most_frequent_quarter']['dev_accuracy_pct']:.2f}"
            )
    for label, m in means.items():
        print(
            f"{label} 3-seed mean: selection {m['selection_score']['mean']:.4f} "
            f"(SD {m['selection_score']['sample_sd']:.2f}), intent "
            f"{m['intent_accuracy_pct']['mean']:.2f}, slot {m['slot_micro_f1_pct']['mean']:.2f}"
        )
    print(json.dumps(rules, indent=2))


if __name__ == "__main__":
    main()
