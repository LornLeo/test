"""Role D: build and execute the Stage 2 notebook for Role C's proposals.

The notebook embeds the full source snapshot (as in main.ipynb), runs the tests,
retrains the frozen Stage 2 configuration from results/stage2_c_final/freeze.json,
stops if the retrained Dev scores differ from the freeze record, then reloads the
best checkpoint and evaluates Dev and Test once. Outputs are kept in the notebook.

Usage (from the repository root, with the official data and GloVe archive local):
    python -m scripts.d_build_stage2_notebook
"""

import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path.cwd().resolve()
NOTEBOOK = ROOT / "notebooks/D_stage2.ipynb"
FREEZE = ROOT / "results/stage2_c_final/freeze.json"
SOURCE_FILES = [
    "src/__init__.py",
    "src/data/annotations.py",
    "src/data/labels.py",
    "src/data/loader.py",
    "src/evaluation/metrics.py",
    "src/model/__init__.py",
    "src/model/artifacts.py",
    "src/model/crf.py",
    "src/model/data.py",
    "src/model/evaluate.py",
    "src/model/model.py",
    "src/model/predict.py",
    "src/model/stage2.py",
    "src/model/tokenization.py",
    "src/model/train.py",
    "scripts/__init__.py",
    "scripts/check_a_integration.py",
    "scripts/prepare_glove.py",
    "tests/test_annotations.py",
    "tests/test_artifacts.py",
    "tests/test_integration.py",
    "tests/test_labels.py",
    "tests/test_metrics.py",
    "tests/test_model.py",
    "tests/test_stage2.py",
    # Text fixture used by tests/test_stage2.py (the Stage 1 model configuration).
    "results/b_glove100_char_crf_wd_slot2_s42/config.json",
]

INTRO = """# Stage 2 system: AI-assisted improvement from Role C's proposals (Role D)

**Stage 2 = Stage 1 recipe + {idea_short}.**

**The Stage 1 system** is B's frozen weight-2 model (shared BiLSTM, character CNN, CRF slot decoder, slot loss weight 2). It has 907,002 parameters, and its Dev and Test results are in `results/final_weight2/`.

**How Stage 2 was made:**
- **Proposals, with AI:** Role C's AI-assisted analysis of Stage 1 ("Part C") proposed two experiments. C1 (primary): class-weighted intent cross-entropy, so that rare intents count more. C2 (secondary): a longer training budget.
- **The rules,** committed before the runs (report Section 4):
  - every change is compared with Stage 1 retrained on the same machine, on Dev only;
  - C1 is confirmed on 3 seeds;
  - C2 (C1 trained for up to 100 epochs, patience 12) must beat C1 on seed 42 before more seeds are run;
  - Stage 2 is C1 unless C2 has the higher 3-seed mean;
  - seed 42 is the representative seed.
- **The outcome** is in report Section 5.3 and `results/stage2_c_final/freeze.json`.
- **Test:** used only once, in the last cell, after the freeze. Nothing is changed after it.

**What this notebook does:**
1. Embeds and tests the complete source.
2. Retrains the frozen configuration from scratch.
3. Stops if the retrained Dev scores differ from the freeze record.
4. Reloads the best checkpoint without further training, and evaluates Dev and Test."""

ARCHITECTURE = """## Stage 2 model

| Part | Stage 1 | Stage 2 |
|---|---|---|
| Input | Raw `utt`, whitespace tokens | Same |
| Word vectors | 100d GloVe rows for Train-vocabulary words, trainable; other words become `<unk>` | Same |
| Character features | 16d character embeddings, CNN (64 channels, width 3), max-pooled per word | Same |
| Encoder | 1-layer BiLSTM, 128 hidden units per direction, dropout 0.3, word dropout 0.1 | Same |
| Intent head | Masked mean + max pooling → linear layer (60 intents) | Same |
| Slot head | Linear layer → BIO-constrained CRF (111 tags, 55 slot types) | Same |
| Loss | Intent cross-entropy + 2 × CRF negative log-likelihood per token | **Intent cross-entropy with class weights** from Train counts: (median / count)^0.5, clipped to [0.25, 4.0], average weight 1 over Train examples (range 0.53–5.39). Training loss only. Slot loss unchanged |
| Optimizer | AdamW (learning rate 0.001, weight decay 1e-05), ReduceLROnPlateau (factor 0.5, patience 2), batch size 64 | Same |
| Model selection | Best epoch by mean of Dev intent accuracy and Dev slot micro-F1; early stopping (patience 8, at most 60 epochs) | {budget} |

The parameter count is unchanged: class weights are not parameters."""

SETUP = """import json, os, sys, tempfile, hashlib
from pathlib import Path
RESOURCE_ROOT = Path.cwd().resolve()
WORK = Path(tempfile.mkdtemp(prefix="sc4002-d-stage2-c-"))
SOURCE_FILES = json.loads({files!r})
for relative in SOURCE_FILES:
    (WORK / relative).parent.mkdir(parents=True, exist_ok=True)
for name in list(sys.modules):
    if name in ("src", "scripts") or name.startswith(("src.", "scripts.")):
        del sys.modules[name]
sys.path.insert(0, str(WORK))
print("Resource root:", RESOURCE_ROOT)
print("Embedded files:", len(SOURCE_FILES))"""

SNAPSHOT = """snapshot = {{name: (WORK / name).read_text(encoding="utf-8") for name in SOURCE_FILES}}
snapshot_sha = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
assert snapshot_sha == {sha!r}
print("Verified embedded source SHA-256:", snapshot_sha)"""

TESTS = """import unittest
suite = unittest.defaultTestLoader.discover(str(WORK / "tests"))
result = unittest.TextTestRunner(verbosity=1).run(suite)
assert result.wasSuccessful()"""

RESOURCES = """from src.model.train import file_sha256
from src.data.loader import load_massive
from scripts.prepare_glove import prepare

DATA = RESOURCE_ROOT / "data/1.0/data/en-US.jsonl"
assert DATA.exists(), "Download MASSIVE 1.0 first (README.md, 'Data download and verification')"
assert file_sha256(DATA) == "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e"
splits = load_massive(DATA)
assert (len(splits["train"]), len(splits["dev"]), len(splits["test"])) == (11514, 2033, 2974)
print("Verified official English data:", DATA)

GLOVE = RESOURCE_ROOT / "embeddings/glove-wiki-gigaword-100.gz"
manifest = prepare(GLOVE, WORK / "embeddings/glove_train_100.txt", DATA, download=True)
assert manifest["source_sha256"] == "4934d4708bf3d65f28fc984dbbfa18af9cbd2505d166a7258a1a957808559b43"
print("Verified public GloVe archive:", GLOVE)"""

SETTINGS = """FREEZE = json.loads({freeze!r})
from src.model.train import make_parser
ACCEPTED = set(vars(make_parser().parse_args([])))
SETTINGS = {{k: v for k, v in FREEZE["arguments"].items() if k in ACCEPTED}}
SETTINGS.update(
    data=str(DATA),
    word_vectors=str(WORK / "embeddings/glove_train_100.txt"),
    run_dir=str(RESOURCE_ROOT / "results/stage2_c_final/notebook_run"),
    checkpoint=str(RESOURCE_ROOT / "checkpoints/stage2_c/d_stage2_c_s42_notebook.pt"),
)
changed = {{k for k in SETTINGS if SETTINGS[k] != FREEZE["arguments"][k]}}
assert changed == {{"data", "word_vectors", "run_dir", "checkpoint"}}, changed
assert SETTINGS["intent_class_weights"] is True
print(json.dumps(SETTINGS, indent=2))"""

TRAIN = """from argparse import Namespace
from src.model.train import train
summary = train(Namespace(**SETTINGS))
assert summary["reload_verified"] and not summary["test_evaluated"]
print(json.dumps(summary, indent=2))
print(
    f"Best checkpoint: epoch {summary['best_epoch']} of {summary['epochs_run']}, "
    f"Dev selection score {summary['selection_score']:.2f}% "
    f"(intent {summary['dev_metrics']['intent_accuracy_pct']:.2f}%, "
    f"slot micro-F1 {summary['dev_metrics']['slot_micro_f1_pct']:.2f}%), "
    f"file {Path(summary['checkpoint']).name}"
)"""

CURVES = """from IPython.display import display, Image
display(Image(filename=str(Path(SETTINGS["run_dir"]) / "learning_curves.png")))"""

MODEL = """import torch
from src.model.model import JointBiLSTM, ModelConfig, parameter_counts
payload = torch.load(SETTINGS["checkpoint"], map_location="cpu", weights_only=True)
model = JointBiLSTM(ModelConfig(**payload["model_config"]))
model.load_state_dict(payload["model_state"])
print(model)
counts = parameter_counts(model)
assert counts["total"] <= 20_000_000
print(f"Total parameters: {counts['total']:,}; trainable: {counts['trainable']:,}; cap 20,000,000")
run_config = json.loads((Path(SETTINGS["run_dir"]) / "config.json").read_text(encoding="utf-8"))
print("Intent class weights:", json.dumps(run_config["intent_class_weights"]))"""

REPRO = """from src.model.artifacts import model_state_sha256
assert summary["dev_metrics"] == FREEZE["dev_metrics"], (
    "Retrained Dev scores differ from the frozen selection run; stop and report, do not reselect"
)
assert summary["best_epoch"] == FREEZE["best_epoch"]
assert summary["parameters"] == FREEZE["parameters"]
notebook_tensors = model_state_sha256(payload["model_state"])
frozen_checkpoint = RESOURCE_ROOT / FREEZE["checkpoint"]
if frozen_checkpoint.exists():
    frozen_state = torch.load(frozen_checkpoint, map_location="cpu", weights_only=True)["model_state"]
    print("Tensors identical to the frozen selection run:", notebook_tensors == model_state_sha256(frozen_state))
else:
    print("Frozen selection checkpoint not available on this machine; Dev equality checked above")
print("Retrained Dev scores equal the freeze record:", summary["dev_metrics"])"""

FINAL = """from datetime import datetime
from zoneinfo import ZoneInfo
from src.model.evaluate import evaluate_checkpoint
from src.model.train import write_json
from IPython.display import Markdown

FINAL_OUTPUT = RESOURCE_ROOT / "results/stage2_c_final/notebook"
checkpoint = Path(SETTINGS["checkpoint"])
freeze = {
    "frozen_at_sgt": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
    "checkpoint": checkpoint.name,
    "checkpoint_sha256": file_sha256(checkpoint),
    "model_state_sha256": notebook_tensors,
    "selection_freeze_record": "results/stage2_c_final/freeze.json",
    "training_arguments": FREEZE["arguments"],
    "representative_seed": SETTINGS["seed"],
    "best_epoch": summary["best_epoch"],
    "epochs_run": summary["epochs_run"],
    "dev_metrics": summary["dev_metrics"],
    "selection_rule": "Mean of Dev intent accuracy and exact-span slot micro-F1",
    "test_used_for_selection": False,
}
FINAL_OUTPUT.mkdir(parents=True, exist_ok=False)  # refuses a second final evaluation
write_json(FINAL_OUTPUT / "freeze.json", freeze)
dev_final = evaluate_checkpoint(checkpoint, DATA, "dev", FINAL_OUTPUT / "dev")
assert dev_final["intent_accuracy_pct"] == summary["dev_metrics"]["intent_accuracy_pct"]
assert dev_final["slot_micro_f1_pct"] == summary["dev_metrics"]["slot_micro_f1_pct"]
test_final = evaluate_checkpoint(checkpoint, DATA, "test", FINAL_OUTPUT / "test", frozen=True)
assert file_sha256(checkpoint) == freeze["checkpoint_sha256"]
final_result = {"freeze": freeze, "summary": summary, "dev": dev_final, "test": test_final}
write_json(FINAL_OUTPUT / "final_results.json", final_result)
table = (
    "| Model version | Split | Total parameters | Trainable parameters | Intent accuracy (%) "
    "| Slot micro-F1 (%) | Epochs run / best epoch |\\n|---|---|---:|---:|---:|---:|---:|\\n"
)
for split, metrics in (("Validation", dev_final), ("Test", test_final)):
    table += (
        f"| Stage-2 system | {split} | {metrics['parameters']['total']:,} "
        f"| {metrics['parameters']['trainable']:,} | {metrics['intent_accuracy_pct']:.2f} "
        f"| {metrics['slot_micro_f1_pct']:.2f} | {summary['epochs_run']} / {summary['best_epoch']} |\\n"
    )
display(Markdown(table))
print("D_STAGE2_C_FINAL_RESULT=" + json.dumps({"dev": dev_final, "test": test_final}))"""


def describe(freeze):
    long = freeze["configuration"] == "C2"
    return {
        "idea_short": "C1 (class-weighted intent loss)"
        + (" + C2 (100 epochs, patience 12)" if long else ""),
        "budget": "Same rule; **at most 100 epochs, patience 12** (C2)" if long else "Same",
    }


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(text):
    return nbformat.v4.new_code_cell(text)


def build():
    if NOTEBOOK.exists():
        raise FileExistsError(f"Refusing to overwrite an executed notebook: {NOTEBOOK}")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    # src/ and scripts/ are namespace packages in the repository; as in main.ipynb, the
    # notebook writes empty __init__.py files for them.
    empty_init = {"src/__init__.py", "scripts/__init__.py"}
    snapshot = {
        name: "\n"
        if name in empty_init and not (ROOT / name).exists()
        else (ROOT / name).read_text(encoding="utf-8")
        for name in SOURCE_FILES
    }
    sha = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    cells = [
        md(INTRO.format(**describe(freeze))),
        code(
            "import sys, torch, numpy\n"
            'print("Python:", sys.version)\n'
            'print("PyTorch:", torch.__version__, "NumPy:", numpy.__version__)'
        ),
        md("## Embedded source and tests\n\nThe complete source snapshot is stored below."),
        code(SETUP.format(files=json.dumps(SOURCE_FILES))),
    ]
    for name in SOURCE_FILES:
        cells.append(md(f"### `{name}`"))
        cells.append(code(f"%%writefile {{WORK}}/{name}\n{snapshot[name]}"))
    cells += [
        code(SNAPSHOT.format(sha=sha)),
        code(TESTS),
        md("## Official resources"),
        code(RESOURCES),
        md(
            "## Frozen Stage 2 settings\n\nFrom `results/stage2_c_final/freeze.json`; only paths change."
        ),
        code(SETTINGS.format(freeze=json.dumps(freeze))),
        md(ARCHITECTURE.format(**describe(freeze))),
        md(
            "## Training and model selection (AI-enhanced checkpoint)\n\n"
            "Each epoch prints train and Dev loss, Dev intent accuracy, slot micro-F1, the "
            "selection mean and time. The best Dev selection score is checkpointed."
        ),
        code(TRAIN),
        code(CURVES),
        code(MODEL),
        md(
            "## Reproduction check\n\nThe retrained run must equal the frozen selection run "
            "before Test is evaluated."
        ),
        code(REPRO),
        md(
            "## Final evaluation of the AI-enhanced checkpoint\n\n"
            "The best checkpoint is reloaded without further training. Dev must equal the "
            "selection score above; Test is evaluated once."
        ),
        code(FINAL),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    client = NotebookClient(
        notebook, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    client.execute()
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK)
    print("Executed and saved", NOTEBOOK)


if __name__ == "__main__":
    build()
