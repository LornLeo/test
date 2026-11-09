"""Train on official Train, select only on Dev, and reload the best checkpoint."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loader import load_massive
from src.evaluation.metrics import evaluate
from src.model.data import IGNORE_INDEX, JointDataset, collate_supervised, fit_metadata
from src.model.model import (
    JointBiLSTM,
    ModelConfig,
    initialize_word_vectors,
    parameter_counts,
)
from src.model.predict import JointPredictor, decode_batch
from src.model.stage2 import intent_class_weights


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, threads: int = 4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def select_device(requested: str) -> torch.device:
    # Packed recurrent models use the well-supported CPU/CUDA path by default.
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def joint_losses(logits, batch, device, slot_weight: float = 1.0, model=None, intent_weights=None):
    intent_logits, slot_logits = logits
    intent_targets = batch["intent_labels"].to(device)
    slot_targets = batch["slot_labels"].to(device)
    intent_sum = nn.functional.cross_entropy(
        intent_logits, intent_targets, weight=intent_weights, reduction="sum"
    )
    tokens = int((slot_targets != IGNORE_INDEX).sum())
    if model is not None and model.config.decoder == "crf":
        slot_sum = model.crf.neg_log_likelihood(
            slot_logits, slot_targets, slot_targets != IGNORE_INDEX
        ).sum()
    else:
        slot_sum = nn.functional.cross_entropy(
            slot_logits.reshape(-1, slot_logits.shape[-1]),
            slot_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
    loss = intent_sum / len(intent_targets) + slot_weight * slot_sum / tokens
    return loss, float(intent_sum.detach()), float(slot_sum.detach()), tokens


def run_epoch(
    model, loader, device, metadata, slot_weight=1.0, optimizer=None, intent_weights=None
):
    training = optimizer is not None
    model.train(training)
    intent_total = slot_total = 0.0
    sample_count = token_count = 0
    predictions = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["input_ids"].to(device), batch["lengths"], batch["char_ids"].to(device)
            )
            loss, intent_sum, slot_sum, tokens = joint_losses(
                logits, batch, device, slot_weight, model, intent_weights if training else None
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; run aborted without reporting a score")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            else:
                predictions.extend(decode_batch(batch, *logits, metadata, model))
            intent_total += intent_sum
            slot_total += slot_sum
            sample_count += len(batch["examples"])
            token_count += tokens
    intent_loss, slot_loss = intent_total / sample_count, slot_total / token_count
    return {
        "loss": intent_loss + slot_weight * slot_loss,
        "intent_loss": intent_loss,
        "slot_loss": slot_loss,
    }, predictions


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def save_training_state(path, model, optimizer, scheduler, generator, **progress):
    numpy_state = np.random.get_state()
    payload = {
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "generator_state": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        **progress,
    }
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_training_state(payload, model, optimizer, scheduler, generator):
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    generator.set_state(payload["generator_state"].cpu())
    torch.set_rng_state(payload["torch_rng"].cpu())
    random.setstate(payload["python_rng"])
    state = payload["numpy_rng"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    if payload["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])


def write_history(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, dataset: JointDataset, predictions: list[dict]):
    with path.open("w", encoding="utf-8") as stream:
        for ex, truth, prediction in zip(dataset.examples, dataset.gold, predictions):
            if truth["id"] != prediction["id"]:
                raise ValueError("Cannot export misordered predictions")
            stream.write(
                json.dumps(
                    {"id": ex.sample_id, "utt": ex.utt, "gold": truth, "prediction": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )


def plot_history(rows: list[dict], output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for key, label in (("train_loss", "Train"), ("dev_loss", "Dev")):
        axes[0].plot(epochs, [row[key] for row in rows], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Joint loss", title="Learning curves")
    for key, label in (
        ("intent_accuracy_pct", "Intent accuracy"),
        ("slot_micro_f1_pct", "Slot micro-F1"),
        ("selection_score", "Selection mean"),
    ):
        axes[1].plot(epochs, [row[key] for row in rows], label=label)
    axes[1].set(xlabel="Epoch", ylabel="Dev score (%)", title="Validation only")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def train(args) -> dict:
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, patience, batch_size and threads must be positive")
    if args.slot_weight <= 0 or args.learning_rate <= 0:
        raise ValueError("Loss weight and learning rate must be positive")
    if (
        args.layers < 1
        or args.hidden_dim < 1
        or args.embedding_dim < 1
        or not 0 <= args.dropout < 1
        or not 0 <= args.word_dropout < 1
    ):
        raise ValueError("Invalid model dimensions, layers, or dropout")
    if args.freeze_embeddings and not args.word_vectors:
        raise ValueError("Freezing randomly initialized embeddings is not supported")
    if not 0 <= args.cuda_memory_fraction <= 1:
        raise ValueError("cuda_memory_fraction must be between zero and one")
    output = Path(args.run_dir)
    if args.resume:
        if not output.is_dir():
            raise ValueError("Resume requires the existing run directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log = (output / "training.log").open("a" if args.resume else "w", buffering=1)

    def emit(message):
        print(message, flush=True)
        print(message, file=log, flush=True)

    seed_everything(args.seed, args.threads)
    device = select_device(args.device)
    if device.type == "cuda" and args.cuda_memory_fraction:
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, index)
    started = time.perf_counter()
    splits = load_massive(args.data)
    # Test annotations never enter fitting, model selection, or dev diagnostics.
    metadata = fit_metadata(splits["train"], args.min_frequency)
    train_data = JointDataset(splits["train"], metadata)
    dev_data = JointDataset(splits["dev"], metadata)
    config = ModelConfig(
        len(metadata["vocab"]),
        len(metadata["intent_to_id"]),
        len(metadata["bio_to_id"]),
        args.embedding_dim,
        args.hidden_dim,
        args.layers,
        args.dropout,
        args.freeze_embeddings,
        char_vocab_size=len(metadata["char_vocab"]) if args.char_features else 0,
        decoder=args.decoder,
        bio_tags=list(metadata["bio_to_id"]),
        word_dropout=args.word_dropout,
        intent_pooling=getattr(args, "intent_pooling", "mean_max"),
    )
    model = JointBiLSTM(config)
    vectors = (
        initialize_word_vectors(model, metadata["vocab"], args.word_vectors)
        if args.word_vectors
        else {"source": "random initialization"}
    )
    counts = parameter_counts(model)
    intent_weights = None
    if getattr(args, "intent_class_weights", False):
        # Stage 2 (Role C's proposal C1): Train labels only; training loss only.
        intent_weights = intent_class_weights(splits["train"], metadata["intent_to_id"]).to(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_supervised,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_supervised
    )
    data_sha = file_sha256(args.data)
    run_config = {
        "arguments": vars(args),
        "model": asdict(config),
        "parameters": counts,
        "word_vectors": vectors,
        "intent_class_weights": None
        if intent_weights is None
        else {
            "formula": "(median/count)^0.5 clipped to [0.25, 4.0], Train-example mean 1",
            "min": float(intent_weights.min()),
            "max": float(intent_weights.max()),
            "applies_to": "training loss only",
        },
        "data_sha256": data_sha,
        "selection": "mean of Dev intent_accuracy_pct and slot_micro_f1_pct",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
    }
    emit(json.dumps(run_config, ensure_ascii=False))
    emit(
        f"Gold BIO round-trip: Train={len(train_data)}/{len(train_data)}, Dev={len(dev_data)}/{len(dev_data)}"
    )
    emit(
        f"Parameters: total={counts['total']:,}; trainable={counts['trainable']:,}; cap=20,000,000"
    )
    rows, best_score, best_epoch, first_epoch, prior_seconds = [], -1.0, 0, 1, 0.0
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ("model", "data_sha256"):
            previous_value = previous["run_config"][key]
            if key == "model":
                previous_value = asdict(ModelConfig(**previous_value))
            if previous_value != run_config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        mutable = {"epochs", "resume"}
        for key, value in run_config["arguments"].items():
            defaults = {
                "word_dropout": 0.0,
                "cuda_memory_fraction": 0.0,
                "intent_pooling": "mean_max",
                "intent_class_weights": False,
            }
            if (
                key not in mutable
                and previous["run_config"]["arguments"].get(key, defaults.get(key)) != value
            ):
                raise ValueError(f"Resume argument mismatch: {key}")
        restore_training_state(previous, model, optimizer, scheduler, generator)
        rows, best_score, best_epoch = (
            previous["rows"],
            previous["best_score"],
            previous["best_epoch"],
        )
        first_epoch, prior_seconds = previous["epoch"] + 1, previous["elapsed_seconds"]
        if not checkpoint.exists():
            raise ValueError("Original best checkpoint is missing")
        emit(f"Resuming after epoch {first_epoch - 1}; original best epoch={best_epoch}")
    write_json(output / "config.json", run_config)
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_losses, _ = run_epoch(
            model, train_loader, device, metadata, args.slot_weight, optimizer, intent_weights
        )
        dev_losses, predictions = run_epoch(model, dev_loader, device, metadata, args.slot_weight)
        metrics = evaluate(predictions, dev_data.gold)
        score = (metrics["intent_accuracy_pct"] + metrics["slot_micro_f1_pct"]) / 2
        row = {
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "train_intent_loss": train_losses["intent_loss"],
            "train_slot_loss": train_losses["slot_loss"],
            "dev_loss": dev_losses["loss"],
            "dev_intent_loss": dev_losses["intent_loss"],
            "dev_slot_loss": dev_losses["slot_loss"],
            **metrics,
            "selection_score": score,
            "learning_rate": learning_rate,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "elapsed_seconds": prior_seconds + time.perf_counter() - started,
        }
        rows.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            payload = {
                "model_state": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "model_config": asdict(config),
                "metadata": metadata,
                "epoch": epoch,
                "selection_score": score,
                "dev_metrics": metrics,
                "data_sha256": data_sha,
                "run_config": run_config,
            }
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint)
            write_predictions(output / "dev_predictions.jsonl", dev_data, predictions)
        write_history(output / "history.csv", rows)
        emit(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} dev_loss={row['dev_loss']:.4f} "
            f"dev_intent={metrics['intent_accuracy_pct']:.2f}% dev_slot_F1={metrics['slot_micro_f1_pct']:.2f}% "
            f"mean={score:.2f}% time={row['epoch_seconds']:.1f}s best_epoch={best_epoch}"
        )
        scheduler.step(score)
        save_training_state(
            output / "last_training_state.pt",
            model,
            optimizer,
            scheduler,
            generator,
            epoch=epoch,
            rows=rows,
            best_score=best_score,
            best_epoch=best_epoch,
            elapsed_seconds=row["elapsed_seconds"],
            run_config=run_config,
        )
        if epoch - best_epoch >= args.patience:
            emit(f"Early stopping: no Dev selection improvement for {args.patience} epochs")
            break
    # Fresh object and raw text only; no training after loading the selected checkpoint.
    predictor = JointPredictor(checkpoint, str(device))
    predictions = predictor.predict([record["utt"] for record in splits["dev"]], args.batch_size)
    for prediction, record in zip(predictions, splits["dev"]):
        prediction["id"] = record["id"]
    reloaded = evaluate(predictions, dev_data.gold)
    expected = predictor.checkpoint_info["dev_metrics"]
    if reloaded != expected:
        raise RuntimeError(f"Reloaded Dev scores differ: saved={expected}, reloaded={reloaded}")
    summary = {
        "model_version": "B joint BiLSTM",
        "split": "dev",
        "parameters": counts,
        "best_epoch": best_epoch,
        "epochs_run": len(rows),
        "selection_score": best_score,
        "dev_metrics": reloaded,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "elapsed_seconds": prior_seconds + time.perf_counter() - started,
        "reload_verified": True,
        "test_evaluated": False,
    }
    write_json(output / "summary.json", summary)
    plot_history(rows, output / "learning_curves.png")
    emit("BEST CHECKPOINT: " + json.dumps(summary, ensure_ascii=False))
    log.close()
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/1.0/data/en-US.jsonl")
    parser.add_argument("--run-dir", default="results/b_joint_v1")
    parser.add_argument("--checkpoint", default="checkpoints/b_joint_v1.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--slot-weight", type=float, default=1.0)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--word-vectors")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--char-features", action="store_true")
    parser.add_argument("--decoder", choices=("softmax", "constrained", "crf"), default="softmax")
    parser.add_argument("--word-dropout", type=float, default=0.0)
    parser.add_argument("--intent-pooling", choices=("mean_max", "attention"), default="mean_max")
    parser.add_argument(
        "--intent-class-weights",
        action="store_true",
        help="Stage 2 (Role C's proposal C1): class-weighted intent loss from Train counts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.0,
        help="Optional CUDA allocator cap as a fraction of one visible GPU's memory",
    )
    parser.add_argument(
        "--resume", help="Resume an epoch-boundary last_training_state.pt in the same run directory"
    )
    return parser


if __name__ == "__main__":
    train(make_parser().parse_args())
