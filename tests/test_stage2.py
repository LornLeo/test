"""Stage 2 (Role D, Role C's proposals): defaults reproduce Stage 1; C1 behaves as declared."""

import json
import unittest
from pathlib import Path

import torch

from src.model.data import JointDataset, collate_supervised, fit_metadata
from src.model.model import JointBiLSTM, ModelConfig, parameter_counts
from src.model.stage2 import intent_class_weights
from src.model.train import joint_losses, make_parser

ROOT = Path(__file__).resolve().parents[1]
STAGE1_CONFIG = ROOT / "results/b_glove100_char_crf_wd_slot2_s42/config.json"
STAGE1_EXPORT = ROOT / "artifacts/checkpoints/b_weight2_inference.pt"

# Two existing official MASSIVE Train records, not added to the dataset.
RECORDS = [
    {
        "partition": "train",
        "id": "1",
        "intent": "alarm_set",
        "utt": "wake me up at nine am on friday",
        "annot_utt": "wake me up at [time : nine am] on [date : friday]",
    },
    {
        "partition": "train",
        "id": "2",
        "intent": "alarm_set",
        "utt": "set an alarm for two hours from now",
        "annot_utt": "set an alarm for [time : two hours from now]",
    },
]


class TestStage1Unchanged(unittest.TestCase):
    def test_parser_default_is_stage1(self):
        self.assertFalse(make_parser().parse_args([]).intent_class_weights)

    def test_stage1_config_parameter_count(self):
        config = json.loads(STAGE1_CONFIG.read_text(encoding="utf-8"))["model"]
        model = JointBiLSTM(ModelConfig(**config))
        self.assertEqual(parameter_counts(model), {"total": 907_002, "trainable": 907_002})

    @unittest.skipUnless(STAGE1_EXPORT.exists(), "frozen Stage 1 export not present")
    def test_frozen_stage1_weights_load_strictly(self):
        payload = torch.load(STAGE1_EXPORT, map_location="cpu", weights_only=True)
        model = JointBiLSTM(ModelConfig(**payload["model_config"]))
        model.load_state_dict(payload["model_state"], strict=True)


class TestIntentClassWeights(unittest.TestCase):
    """Role C's proposal C1: weights from Train only, training loss only."""

    def records(self):
        counts = {"alarm_set": 8, "alarm_query": 2, "weather_query": 32}
        return [
            {"partition": "train", "intent": intent, "utt": "x", "annot_utt": "x"}
            for intent, n in counts.items()
            for _ in range(n)
        ]

    def test_weights_follow_declared_formula(self):
        intent_to_id = {"alarm_query": 0, "alarm_set": 1, "weather_query": 2}
        weights = intent_class_weights(self.records(), intent_to_id)
        counts = torch.tensor([2.0, 8.0, 32.0])
        raw = (8.0 / counts).sqrt().clamp(0.25, 4.0)
        expected = raw * counts.sum() / (raw * counts).sum()
        self.assertTrue(torch.allclose(weights, expected))
        self.assertAlmostEqual(float((weights * counts).sum() / counts.sum()), 1.0, places=6)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_rejects_non_train_records(self):
        records = self.records()
        records[0] = dict(records[0], partition="dev")
        with self.assertRaises(ValueError):
            intent_class_weights(records, {"alarm_query": 0, "alarm_set": 1, "weather_query": 2})

    def test_uniform_weights_match_unweighted_loss(self):
        metadata = fit_metadata(RECORDS)
        batch = collate_supervised(list(JointDataset(RECORDS, metadata)))
        model = JointBiLSTM(
            ModelConfig(
                len(metadata["vocab"]),
                len(metadata["intent_to_id"]),
                len(metadata["bio_to_id"]),
                embedding_dim=8,
                hidden_dim=8,
            )
        )
        logits = model(batch["input_ids"], batch["lengths"], batch["char_ids"])
        plain = joint_losses(logits, batch, "cpu", 1.0, model)[0]
        ones = torch.ones(len(metadata["intent_to_id"]))
        weighted = joint_losses(logits, batch, "cpu", 1.0, model, ones)[0]
        self.assertTrue(torch.allclose(plain, weighted))

    def test_weights_change_the_training_loss(self):
        # Two intents are needed: with one class the intent loss is always zero.
        records = [RECORDS[0], dict(RECORDS[1], intent="alarm_query")]
        metadata = fit_metadata(records)
        batch = collate_supervised(list(JointDataset(records, metadata)))
        model = JointBiLSTM(
            ModelConfig(
                len(metadata["vocab"]),
                len(metadata["intent_to_id"]),
                len(metadata["bio_to_id"]),
                embedding_dim=8,
                hidden_dim=8,
            )
        )
        logits = model(batch["input_ids"], batch["lengths"], batch["char_ids"])
        plain = joint_losses(logits, batch, "cpu", 1.0, model)[0]
        doubled = torch.full((len(metadata["intent_to_id"]),), 2.0)
        weighted = joint_losses(logits, batch, "cpu", 1.0, model, doubled)[0]
        self.assertFalse(torch.allclose(plain, weighted))


if __name__ == "__main__":
    unittest.main()
