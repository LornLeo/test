"""Role D: evidence on Role C's class-weighting proposal (C1), from Train and Dev only.

Measures how uneven the Train intent distribution is, whether rare intents score
worse on Dev for the Stage 1 system, and the predeclared class weights. Writes
results/role_d/intent_imbalance.json. No training; Test is never read.

Usage (from the repository root):
    python -m scripts.d_intent_imbalance
"""

import contextlib
import csv
import io
import json
from collections import Counter
from pathlib import Path

from src.data.loader import load_massive
from src.model.data import fit_metadata
from src.model.stage2 import (
    INTENT_WEIGHT_CLIP,
    INTENT_WEIGHT_POWER,
    intent_class_weights,
)

STAGE1_RUN = "b_glove100_char_crf_wd_slot2_s42"
OUTPUT = Path("results/role_d/intent_imbalance.json")


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2
        i = j + 1
    return result


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def quarter_accuracy(rows, train_counts):
    order = sorted(rows, key=lambda r: (train_counts[r["intent"]], r["intent"]))
    q = len(order) // 4
    result = {}
    for name, group in (("rarest_quarter", order[:q]), ("most_frequent_quarter", order[-q:])):
        support = sum(int(r["gold_support"]) for r in group)
        correct = sum(int(r["correct"]) for r in group)
        result[name] = {
            "intents": len(group),
            "train_count_range": [
                min(train_counts[r["intent"]] for r in group),
                max(train_counts[r["intent"]] for r in group),
            ],
            "dev_support": support,
            "dev_accuracy_pct": 100 * correct / support,
        }
    return result


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        train = load_massive("data/1.0/data/en-US.jsonl")["train"]
    counts = Counter(r["intent"] for r in train)
    with open("results/b_handoff/dev_intent_per_class.csv", newline="", encoding="utf-8") as f:
        rows = [
            r for r in csv.DictReader(f) if r["run"] == STAGE1_RUN and int(r["gold_support"]) > 0
        ]
    metadata = fit_metadata(train)
    weights = intent_class_weights(train, metadata["intent_to_id"])
    by_intent = {i: float(weights[k]) for i, k in metadata["intent_to_id"].items()}
    worst = sorted(
        (r for r in rows if int(r["gold_support"]) >= 10), key=lambda r: float(r["f1_pct"])
    )
    frequency_rank = {i: n + 1 for n, (i, _) in enumerate(counts.most_common())}
    result = {
        "source_note": "Train labels and Stage 1 Dev per-class results only; Test unused",
        "stage1_run": STAGE1_RUN,
        "train_intents": len(counts),
        "train_count_min": min(counts.values()),
        "train_count_max": max(counts.values()),
        "train_count_median": sorted(counts.values())[len(counts) // 2],
        "max_to_min_ratio": max(counts.values()) / min(counts.values()),
        "rarest": counts.most_common()[-5:],
        "most_frequent": counts.most_common(5),
        "dev_intents_with_support": len(rows),
        "spearman_train_count_vs_dev_f1": spearman(
            [counts[r["intent"]] for r in rows], [float(r["f1_pct"]) for r in rows]
        ),
        "dev_accuracy_by_frequency": quarter_accuracy(rows, counts),
        "worst_dev_intents_support10": [
            {
                "intent": r["intent"],
                "dev_f1_pct": float(r["f1_pct"]),
                "train_count": counts[r["intent"]],
                "train_frequency_rank": frequency_rank[r["intent"]],
                "class_weight": by_intent[r["intent"]],
            }
            for r in worst[:5]
        ],
        "class_weights": {
            "formula": f"(median/count)^{INTENT_WEIGHT_POWER}, clipped to {list(INTENT_WEIGHT_CLIP)}, "
            "rescaled to a Train-example mean of 1",
            "min": float(weights.min()),
            "max": float(weights.max()),
            "by_intent": dict(sorted(by_intent.items())),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    summary = {k: result[k] for k in ("max_to_min_ratio", "spearman_train_count_vs_dev_f1")}
    print(json.dumps(summary, indent=2))
    print(json.dumps(result["dev_accuracy_by_frequency"], indent=2))
    print(json.dumps(result["worst_dev_intents_support10"], indent=2))
    print("weights min/max", result["class_weights"]["min"], result["class_weights"]["max"])


if __name__ == "__main__":
    main()
