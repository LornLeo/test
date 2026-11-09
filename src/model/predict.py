"""Reloadable predictor whose public input consists only of raw utt strings."""

from pathlib import Path

import torch

from src.model.data import collate_inputs, encode_input
from src.model.model import JointBiLSTM, ModelConfig, parameter_counts
from src.model.tokenization import bio_to_spans


def decode_batch(batch: dict, intent_logits, slot_logits, metadata: dict, model=None) -> list[dict]:
    intents = {index: name for name, index in metadata["intent_to_id"].items()}
    tags = {index: name for name, index in metadata["bio_to_id"].items()}
    intent_ids = intent_logits.argmax(-1).cpu().tolist()
    tag_ids = (
        model.decode_tags(slot_logits, batch["lengths"])
        if model is not None
        else slot_logits.argmax(-1).cpu().tolist()
    )
    predictions = []
    for ex, intent_id, sequence in zip(batch["examples"], intent_ids, tag_ids):
        spans = bio_to_spans(ex.utt, ex.tokens, [tags[t] for t in sequence[: len(ex.tokens)]])
        predictions.append({"id": ex.sample_id, "intent": intents[intent_id], "slots": spans})
    return predictions


class JointPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.metadata = payload["metadata"]
        self.model = JointBiLSTM(ModelConfig(**payload["model_config"]))
        self.model.load_state_dict(payload["model_state"])
        parameter_counts(self.model)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.checkpoint_info = {
            key: payload[key] for key in ("epoch", "dev_metrics", "selection_score", "data_sha256")
        }

    @torch.inference_mode()
    def predict(self, utterances: list[str], batch_size: int = 64) -> list[dict]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        examples = [encode_input(utt, str(i), self.metadata) for i, utt in enumerate(utterances)]
        results = []
        for start in range(0, len(examples), batch_size):
            batch = collate_inputs(examples[start : start + batch_size])
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["lengths"],
                batch["char_ids"].to(self.device),
            )
            results.extend(decode_batch(batch, *logits, self.metadata, self.model))
        # IDs here are positional transport IDs; callers can attach their own IDs.
        return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--utt", action="append", required=True, help="Repeat for multiple raw utterances"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(JointPredictor(args.checkpoint, args.device).predict(args.utt), indent=2))
