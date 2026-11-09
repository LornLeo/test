"""Role D: build and execute the Stage 2 experiment log notebook (Dev only).

The notebook gives main.ipynb evidence for every Stage 2 number that does not come from the
final Stage 2 notebook: the same-machine Stage 1 reference runs, the C1 and C2 runs, the
per-class diagnostics and the design sensitivity sweep.

Only the three submission files reach the markers, so the saved Dev files are embedded in the
notebook with their SHA-256 hashes; the cells then check the hashes and compute every table,
rule outcome and figure from the embedded files. Nothing is typed by hand, nothing is
retrained, and Test is never read.

Usage (from the repository root):
    python -m scripts.d_build_experiment_log
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "D_experiment_log.ipynb"
SEEDS = (42, 43, 44)

CORE_RUNS = {
    "R": [f"results/stage2/d_stage1_ref_s{s}" for s in SEEDS],
    "C1": [f"results/stage2/d_intentcw_s{s}" for s in SEEDS],
    "C2": ["results/stage2/d_intentcw_long_s42"],
}
SWEEP_RUNS = {
    "layers2": [f"results/sweep/d_sweep_layers2_s{s}" for s in SEEDS],
    "layers3": [f"results/sweep/d_sweep_layers3_s{s}" for s in SEEDS],
    "hidden64": [f"results/sweep/d_sweep_hidden64_s{s}" for s in SEEDS],
    "hidden256": [f"results/sweep/d_sweep_hidden256_s{s}" for s in SEEDS],
    "nochar": [f"results/sweep/d_abl_nochar_s{s}" for s in SEEDS],
    "nowd": [f"results/sweep/d_abl_nowd_s{s}" for s in SEEDS],
    "neither": [f"results/sweep/d_abl_neither_s{s}" for s in SEEDS],
}
ANALYSIS = "results/role_d/c_analysis.json"


def read(path):
    text = (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text": text}


def collect():
    files = {ANALYSIS: read(ANALYSIS)}
    for runs in CORE_RUNS.values():
        for run in runs:
            for name in ("summary.json", "history.csv"):
                files[f"{run}/{name}"] = read(f"{run}/{name}")
    sweep_ready = all(
        (ROOT / run / "summary.json").exists()
        for runs in SWEEP_RUNS.values()
        for run in runs
    )
    if sweep_ready:
        for runs in SWEEP_RUNS.values():
            for run in runs:
                for name in ("summary.json", "history.csv"):
                    files[f"{run}/{name}"] = read(f"{run}/{name}")
    for path, entry in files.items():
        if path.endswith("summary.json"):
            summary = json.loads(entry["text"])
            if summary.get("test_evaluated") or not summary.get("reload_verified"):
                raise ValueError(f"{path}: Test evaluated or reload not verified")
    return files, sweep_ready


SETUP = '''import csv, hashlib, io, json, statistics
import matplotlib.pyplot as plt

SEEDS = (42, 43, 44)
CORE_RUNS = __CORE__
SWEEP_RUNS = __SWEEP__
SWEEP_READY = __READY__
EMBEDDED = json.loads(r"""__DATA__""")

for path, entry in EMBEDDED["files"].items():
    assert hashlib.sha256(entry["text"].encode("utf-8")).hexdigest() == entry["sha256"], path
    if path.endswith("summary.json"):
        s = json.loads(entry["text"])
        assert not s["test_evaluated"] and s["reload_verified"], path
print(f"{len(EMBEDDED['files'])} saved Dev files embedded on {EMBEDDED['built_at_utc']} UTC; "
      "all SHA-256 hashes verified; no file records a Test evaluation.")


def summary(run):
    return json.loads(EMBEDDED["files"][f"{run}/summary.json"]["text"])


def history(run):
    rows = csv.DictReader(io.StringIO(EMBEDDED["files"][f"{run}/history.csv"]["text"]))
    return [{k: float(v) for k, v in row.items()} for row in rows]


def row(run):
    s = summary(run)
    return {
        "run": run.split("/")[-1],
        "params": s["parameters"]["total"],
        "intent": s["dev_metrics"]["intent_accuracy_pct"],
        "slot": s["dev_metrics"]["slot_micro_f1_pct"],
        "score": s["selection_score"],
        "best": s["best_epoch"],
        "epochs": s["epochs_run"],
        "minutes": s["elapsed_seconds"] / 60,
    }


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def print_runs(runs):
    print(f"{'run':<24}{'params':>11}{'intent':>9}{'slot F1':>9}{'score':>9}"
          f"{'best/run':>10}{'min':>6}")
    for run in runs:
        r = row(run)
        print(f"{r['run']:<24}{r['params']:>11,}{r['intent']:>9.2f}{r['slot']:>9.2f}"
              f"{r['score']:>9.2f}{r['best']:>6}/{r['epochs']:<3}{r['minutes']:>6.0f}")


def print_means(groups):
    print(f"{'config':<16}{'params':>11}{'intent':>15}{'slot F1':>15}{'score':>15}")
    for name, runs in groups.items():
        rows = [row(r) for r in runs]
        cells = []
        for key in ("intent", "slot", "score"):
            m, sd = mean_sd([r[key] for r in rows])
            cells.append(f"{m:>8.2f} ({sd:.2f})")
        print(f"{name:<16}{rows[0]['params']:>11,}" + "".join(f"{c:>15}" for c in cells))
'''

CORE = """score = {name: [row(r)["score"] for r in runs] for name, runs in CORE_RUNS.items()}
print("All Stage 2 runs on Dev (scores in %; score = mean of intent accuracy and slot F1)\\n")
print_runs([r for runs in CORE_RUNS.values() for r in runs])
print("\\n3-seed means (sample SD):\\n")
print_means({"R (Stage 1)": CORE_RUNS["R"], "C1": CORE_RUNS["C1"]})

print("\\nC1 minus R, by seed (points):",
      ", ".join(f"s{s} {c - r:+.2f}" for s, c, r in zip(SEEDS, score["C1"], score["R"])))
print(f"Mean difference: {statistics.mean(score['C1']) - statistics.mean(score['R']):+.2f}")

rule1 = score["C1"][0] > score["R"][0]
rule2 = score["C2"][0] > score["C1"][0]
stage2 = "C2" if rule2 and statistics.mean(score["C2"]) > statistics.mean(score["C1"]) else "C1"
print(f"\\nRule 1, C1 seed 42 > R seed 42 ({score['C1'][0]:.4f} > {score['R'][0]:.4f}): "
      f"{'passed' if rule1 else 'failed'}")
print(f"Rule 2, C2 seed 42 > C1 seed 42 ({score['C2'][0]:.4f} > {score['C1'][0]:.4f}): "
      f"{'passed' if rule2 else 'failed'}")
print(f"Rule 3, Stage 2 configuration: {stage2} (seed 42 is the submitted checkpoint)")
assert rule1 and not rule2 and stage2 == "C1"
"""

CURVES = """fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, name in zip(axes[:2], ("R", "C1")):
    for run, seed in zip(CORE_RUNS[name], SEEDS):
        h = history(run)
        ax.plot([e["epoch"] for e in h], [e["selection_score"] for e in h], label=f"seed {seed}")
    ax.set_title(f"{name}: Dev selection score per epoch")
    ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
axes[0].set_ylabel("Dev selection score (%)")
c1, c2 = history(CORE_RUNS["C1"][0]), history(CORE_RUNS["C2"][0])
axes[2].plot([e["epoch"] for e in c2], [e["selection_score"] for e in c2], label="C2 seed 42", lw=4, alpha=0.4)
axes[2].plot([e["epoch"] for e in c1], [e["selection_score"] for e in c1], label="C1 seed 42")
axes[2].set_title("C2 vs C1, seed 42"); axes[2].set_xlabel("epoch"); axes[2].grid(alpha=0.3); axes[2].legend()
for ax in axes: ax.set_ylim(80, 85)
plt.tight_layout(); plt.show()

metrics = ("train_loss", "dev_loss", "intent_accuracy_pct", "slot_micro_f1_pct", "selection_score", "learning_rate")
same = all(all(a[k] == b[k] for k in metrics) for a, b in zip(c1, c2))
print(f"C2 repeats C1 exactly for all {len(c1)} C1 epochs: {same}")
print("C2 epochs after C1 stopped:", ", ".join(
    f"{int(e['epoch'])}: {e['selection_score']:.2f} (lr {e['learning_rate']:.1e})" for e in c2[len(c1):]))
assert same
"""

PER_CLASS = """analysis = json.loads(EMBEDDED["files"]["results/role_d/c_analysis.json"]["text"])
print("C1 minus R by seed, Dev (points). Computed from the saved Dev predictions by "
      "scripts/d_c_analysis.py.\\n")
print(f"{'seed':<6}{'rarest quarter':>16}{'most frequent quarter':>24}{'general_quirky F1':>20}{'qa_factoid F1':>16}")
for seed in map(str, SEEDS):
    r, c = analysis["runs"]["R"][seed], analysis["runs"]["C1"][seed]
    rare = c["accuracy_by_train_frequency"]["rarest_quarter"]["dev_accuracy_pct"] - r["accuracy_by_train_frequency"]["rarest_quarter"]["dev_accuracy_pct"]
    freq = c["accuracy_by_train_frequency"]["most_frequent_quarter"]["dev_accuracy_pct"] - r["accuracy_by_train_frequency"]["most_frequent_quarter"]["dev_accuracy_pct"]
    gq = c["watched_intent_f1_pct"]["general_quirky"] - r["watched_intent_f1_pct"]["general_quirky"]
    qa = c["watched_intent_f1_pct"]["qa_factoid"] - r["watched_intent_f1_pct"]["qa_factoid"]
    print(f"{seed:<6}{rare:>+16.2f}{freq:>+24.2f}{gq:>+20.2f}{qa:>+16.2f}")
support = analysis["runs"]["R"]["42"]["accuracy_by_train_frequency"]
print(f"\\nDev support: rarest quarter {support['rarest_quarter']['dev_support']}, "
      f"most frequent quarter {support['most_frequent_quarter']['dev_support']} utterances")
"""

SWEEP = """if not SWEEP_READY:
    print("Sweep runs not finished when this notebook was built; this section is filled in on rebuild.")
else:
    R = CORE_RUNS["R"]
    depth = {1: R, 2: SWEEP_RUNS["layers2"], 3: SWEEP_RUNS["layers3"]}
    width = {64: SWEEP_RUNS["hidden64"], 128: R, 256: SWEEP_RUNS["hidden256"]}
    ablation = {"full recipe": R, "no character CNN": SWEEP_RUNS["nochar"],
                "no word dropout": SWEEP_RUNS["nowd"], "neither": SWEEP_RUNS["neither"]}

    def stats(runs):
        scores = [row(r)["score"] for r in runs]
        m, sd = mean_sd(scores)
        return {"scores": scores, "mean": m, "sd": sd, "params": row(runs[0])["params"]}

    def choose(sweep, current):  # rule 2 of the sweep plan
        s = {k: stats(v) for k, v in sweep.items()}
        best = max(s, key=lambda k: s[k]["mean"])
        beaten = [k for k in s if s[k]["mean"] - s[current]["mean"] > max(s[k]["sd"], s[current]["sd"])]
        near = [k for k in s if s[best]["mean"] - s[k]["mean"] <= max(s[k]["sd"], s[best]["sd"])]
        return s, best, beaten, min(near, key=lambda k: s[k]["params"])

    def plot(s, preferred, xlabel, title, ax, log2=False):
        xs = sorted(s)
        for x in xs:
            ax.scatter([x] * 3, s[x]["scores"], color="0.65", s=18, zorder=2)
        ax.errorbar(xs, [s[x]["mean"] for x in xs], yerr=[s[x]["sd"] for x in xs], marker="o",
                    capsize=4, zorder=3, label="3-seed mean ± SD")
        ax.scatter([preferred], [s[preferred]["mean"]], s=160, facecolors="none", edgecolors="C3",
                   linewidths=2, zorder=4, label=f"preferred = {preferred}")
        if log2:
            ax.set_xscale("log", base=2)
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(xlabel); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for label, sweep, current, ax, xlabel, log2 in (
        ("Depth (BiLSTM layers, width 128)", depth, 1, axes[0], "number of BiLSTM layers", False),
        ("Width (hidden size per direction, 1 layer)", width, 128, axes[1], "hidden size", True),
    ):
        s, best, beaten, preferred = choose(sweep, current)
        print(label)
        for k in sorted(s):
            print(f"  {k:>4}: params {s[k]['params']:>9,}  score {s[k]['mean']:.2f} (SD {s[k]['sd']:.2f})  "
                  f"seeds {', '.join(f'{x:.2f}' for x in s[k]['scores'])}")
        print(f"  best mean = {best}; current ({current}) supported: {not beaten}; preferred = {preferred}\\n")
        plot(s, preferred, xlabel, f"Dev score vs {label.split(' (')[0].lower()}", ax, log2)
    axes[0].set_ylabel("Dev selection score (%)")

    print("Character CNN and word dropout (full recipe minus reduced recipe, by seed)")
    full = stats(ablation["full recipe"])
    for name, reduced in (("character CNN", "no character CNN"), ("word dropout", "no word dropout"),
                          ("both vs neither", "neither")):
        red = stats(ablation[reduced])
        diffs = [f - r for f, r in zip(full["scores"], red["scores"])]
        verdict = ("consistent gain" if all(d > 0 for d in diffs)
                   else "average gain only" if statistics.mean(diffs) > 0 else "no gain")
        print(f"  {name:<16} {', '.join(f'{d:+.2f}' for d in diffs)}  mean {statistics.mean(diffs):+.2f}  -> {verdict}")
    names = list(ablation)
    for i, seed in enumerate(SEEDS):
        axes[2].plot(range(4), [stats(ablation[n])["scores"][i] for n in names], marker="o", label=f"seed {seed}")
    axes[2].plot(range(4), [stats(ablation[n])["mean"] for n in names], marker="s", color="black", lw=2, label="mean")
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels(names, fontsize=8)
    axes[2].set_title("Character CNN and word dropout, 3 seeds"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.show()
"""


def build():
    files, sweep_ready = collect()
    data = {
        "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "files": files,
    }
    payload = json.dumps(data, ensure_ascii=False)
    if '"""' in payload:
        raise ValueError("embedded data must not contain triple quotes")
    setup = (
        SETUP.replace("__CORE__", repr(CORE_RUNS))
        .replace("__SWEEP__", repr(SWEEP_RUNS))
        .replace("__READY__", repr(sweep_ready))
        .replace("__DATA__", payload)
    )
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md(
            "# Stage 2 experiment log (Dev only)\n\n"
            "This section is the evidence for the Stage 2 comparisons quoted in the report that "
            "are not part of the final Stage 2 run above:\n"
            "- the Stage 1 recipe retrained on the same machine (R),\n"
            "- Role C's proposals C1 (class-weighted intent loss) and C2 (longer training),\n"
            "- the per-class diagnostics, and\n"
            "- the design sensitivity sweep (BiLSTM depth and width, character CNN and word "
            "dropout).\n\n"
            "The saved Dev result files of every run were embedded by "
            "`scripts/d_build_experiment_log.py`, with their SHA-256 hashes. The cells below "
            "check the hashes and compute every table, rule outcome and figure from those files. "
            "Nothing is retrained here and **Test is never read**; Test was evaluated once, in "
            "the final Stage 2 cell above."
        ),
        code(setup),
        md(
            "## 1. Same-machine Stage 1 reference (R), C1 and C2\n\n"
            "Rules (fixed before the runs): C1 seed 42 must beat R seed 42 before seeds 43 and "
            "44 are run; C2 seed 42 must beat C1 seed 42; Stage 2 is C1 unless C2 passes and "
            "has the higher 3-seed mean; seed 42 is the submitted checkpoint."
        ),
        code(CORE),
        md("### Learning curves of every run, and C2 against C1"),
        code(CURVES),
        md("## 2. Per-class diagnostics: what C1 traded"),
        code(PER_CLASS),
        md(
            "## 3. Design sensitivity sweep\n\n"
            "Dev only; it does not change the frozen Stage 1 or Stage 2 systems. A depth or "
            "width counts as within reach of the best 3-seed mean if it is no further below it "
            "than the larger of the two seed SDs; the smallest such model is preferred. A "
            "component has a consistent gain if removing it lowers the score on all three seeds."
        ),
        code(SWEEP),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK)
    print(
        f"Executed and saved {NOTEBOOK} ({len(files)} files embedded; sweep ready: {sweep_ready})"
    )


if __name__ == "__main__":
    build()
