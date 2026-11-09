"""Role D: freeze the Stage 2 choice (Role C's proposals) before any Test evaluation.

Recomputes the predeclared decision (scripts/d_c_analysis.py, docs/D_STAGE2_PLAN.md
section 4), checks the representative checkpoint's SHA-256 against its training
record, reloads it on Dev only (src.model.evaluate asserts the reloaded Dev scores
equal the selection scores), and writes results/stage2_c_final/freeze.json.
Never touches Test.

Usage (from the repository root):
    python -m scripts.d_freeze
"""

import contextlib
import dataclasses
import io
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts import d_c_analysis
from src.model.artifacts import model_state_sha256
from src.model.evaluate import evaluate_checkpoint
from src.model.model import ModelConfig
from src.model.train import file_sha256

OUTPUT = Path("results/stage2_c_final")
COMPAT_DIR = Path("checkpoints/stage2_c")
IDEAS = {
    "C1": "Role C's proposal C1: class-weighted intent loss",
    "C2": "Role C's proposals C1 + C2: class-weighted intent loss, 100 epochs, patience 12",
}


def compatible_copy(checkpoint):
    """Checkpoints trained on role-d-stage2 store that branch's unused S1 field
    frozen_vocab_size=0. This branch's ModelConfig has no such field, so the Dev reload
    uses a copy without it. Only that config key is removed; tensors are unchanged."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    extra = {k: v for k, v in payload["model_config"].items() if k not in fields}
    if not extra:
        return checkpoint, None
    if extra != {"frozen_vocab_size": 0}:
        raise ValueError(f"Unexpected model configuration fields: {extra}")
    payload["model_config"] = {k: v for k, v in payload["model_config"].items() if k in fields}
    copy = COMPAT_DIR / f"{checkpoint.stem}_compat.pt"
    COMPAT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, copy)
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"]
    reloaded = torch.load(copy, map_location="cpu", weights_only=True)["model_state"]
    if model_state_sha256(original) != model_state_sha256(reloaded):
        raise ValueError("Compatibility copy changed the model tensors")
    return copy, {
        "removed_model_config_fields": extra,
        "reason": "trained with role-d-stage2 code; S1 option unused (0)",
        "evaluated_copy": str(copy).replace("\\", "/"),
        "evaluated_copy_sha256": file_sha256(copy),
        "model_state_sha256": model_state_sha256(original),
        "tensors_identical": True,
    }


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    freeze_path = OUTPUT / "freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing freeze record: {freeze_path}")
    with contextlib.redirect_stdout(io.StringIO()):
        d_c_analysis.main()
    analysis = d_c_analysis.read_json(d_c_analysis.OUTPUT)
    rules = analysis["rules"]
    if not rules.get("stage2_configuration"):
        raise ValueError("The predeclared rules have not reached a Stage 2 decision yet")
    label, run = rules["stage2_configuration"], rules["stage2_run"]
    run_dir = d_c_analysis.STAGE2 / run
    summary = d_c_analysis.read_json(run_dir / "summary.json")
    config = d_c_analysis.read_json(run_dir / "config.json")
    checkpoint = Path(config["arguments"]["checkpoint"])
    sha = file_sha256(checkpoint)
    if sha != summary["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the training record")
    if summary["test_evaluated"]:
        raise ValueError("Training record reports a Test evaluation")
    evaluated, compatibility = compatible_copy(checkpoint)
    dev = evaluate_checkpoint(
        evaluated, config["arguments"]["data"], "dev", OUTPUT / "freeze_dev_check"
    )
    record = {
        "stage": "Stage 2 (AI-assisted, Role C's proposals), Role D",
        "frozen_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_commit": git("rev-parse", "HEAD"),
        "working_tree_clean_for_src": git("status", "--porcelain", "--", "src") == "",
        "configuration": label,
        "idea": IDEAS[label],
        "representative_seed": 42,
        "run": run,
        "arguments": config["arguments"],
        "intent_class_weights": config.get("intent_class_weights"),
        "model_config": {k: v for k, v in config["model"].items() if k != "bio_tags"},
        "parameters": summary["parameters"],
        "best_epoch": summary["best_epoch"],
        "epochs_run": summary["epochs_run"],
        "selection": config["selection"],
        "dev_selection_score": summary["selection_score"],
        "dev_metrics": summary["dev_metrics"],
        "fresh_dev_reload": {
            k: dev[k] for k in ("intent_accuracy_pct", "slot_micro_f1_pct", "num_examples")
        },
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "checkpoint_sha256": sha,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "dev_reload_compatibility": compatibility,
        "data_sha256": config["data_sha256"],
        "environment": config["environment"],
        "decision": {
            "plan": "docs/D_STAGE2_PLAN.md section 4",
            "analysis": "results/role_d/c_analysis.json",
            "rules": rules,
            "three_seed_means": analysis["three_seed_means"],
        },
        "test_evaluated": False,
        "note": "No configuration change is allowed after this record. Test is evaluated "
        "once, on this checkpoint only, in the Stage 2 notebook.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with freeze_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(
        f"Frozen {run} ({label}): Dev intent {dev['intent_accuracy_pct']:.2f}%, "
        f"slot F1 {dev['slot_micro_f1_pct']:.2f}%, sha256 {sha}"
    )


if __name__ == "__main__":
    main()
