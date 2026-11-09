"""Evaluate a saved model, with an explicit frozen-checkpoint gate for Test."""

import argparse
import json
from pathlib import Path

import torch

from src.data.loader import load_massive
from src.evaluation.metrics import evaluate
from src.model.data import EvaluationDataset
from src.model.model import parameter_counts
from src.model.predict import JointPredictor
from src.model.train import file_sha256, seed_everything, write_json, write_predictions


def evaluate_checkpoint(
    checkpoint,
    data,
    split="dev",
    output="results/evaluation",
    frozen=False,
    batch_size=64,
    device="cpu",
):
    if split not in ("dev", "test"):
        raise ValueError("Final evaluation split must be dev or test")
    if split == "test" and not frozen:
        raise ValueError("Test evaluation requires frozen=True / --frozen; do not use Test to tune")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(42)
    predictor = JointPredictor(checkpoint, device)
    if file_sha256(data) != predictor.checkpoint_info["data_sha256"]:
        raise ValueError("Evaluation data differs from the training run's official data file")
    records = load_massive(data)[split]
    dataset = EvaluationDataset(records, predictor.metadata)
    predictions = predictor.predict([record["utt"] for record in records], batch_size)
    for prediction, record in zip(predictions, records):
        prediction["id"] = record["id"]
    metrics = evaluate(predictions, dataset.gold)
    if split == "dev" and metrics != predictor.checkpoint_info["dev_metrics"]:
        raise RuntimeError("Reloaded Dev scores do not match checkpoint selection scores")
    result = {
        "split": split,
        **metrics,
        "parameters": parameter_counts(predictor.model),
        "best_epoch": predictor.checkpoint_info["epoch"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "frozen_for_test": bool(frozen),
        "num_examples": len(records),
        "torch": str(torch.__version__),
    }
    write_json(output / "metrics.json", result)
    write_predictions(output / "predictions.jsonl", dataset, predictions)
    print(json.dumps(result, indent=2))
    print(
        f"{split}: intent={metrics['intent_accuracy_pct']:.2f}%, slot micro-F1={metrics['slot_micro_f1_pct']:.2f}%"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/1.0/data/en-US.jsonl")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        args.data,
        args.split,
        args.output,
        args.frozen,
        args.batch_size,
        args.device,
    )
