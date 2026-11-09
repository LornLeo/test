"""B-side numerical/contract tests; no generated examples enter real training."""

import itertools
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from src.model.crf import BIOCRF
from src.model.data import (
    EvaluationDataset,
    JointDataset,
    collate_inputs,
    collate_supervised,
    encode_input,
    fit_metadata,
)
from src.model.model import (
    JointBiLSTM,
    MaskedAttentionPooling,
    ModelConfig,
    parameter_counts,
)
from src.model.predict import JointPredictor
from src.model.tokenization import bio_to_spans, spans_to_bio
from src.model.train import (
    joint_losses,
    restore_training_state,
    save_training_state,
    seed_everything,
)

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


class TestJointModel(unittest.TestCase):
    def setUp(self):
        seed_everything(42, 1)
        self.metadata = fit_metadata(RECORDS)
        self.dataset = JointDataset(RECORDS, self.metadata)

    def model(self, chars=False, decoder="softmax", pooling="mean_max"):
        return JointBiLSTM(
            ModelConfig(
                len(self.metadata["vocab"]),
                1,
                len(self.metadata["bio_to_id"]),
                embedding_dim=8,
                hidden_dim=8,
                dropout=0.2,
                char_vocab_size=len(self.metadata["char_vocab"]) if chars else 0,
                char_embedding_dim=4,
                char_channels=6,
                decoder=decoder,
                bio_tags=list(self.metadata["bio_to_id"]),
                intent_pooling=pooling,
            )
        )

    def test_train_only_metadata(self):
        with self.assertRaises(ValueError):
            fit_metadata([dict(RECORDS[0], partition="dev")])

    def test_gold_slot_bio_round_trip(self):
        for ex, truth in zip(self.dataset.examples, self.dataset.gold):
            self.assertEqual(
                bio_to_spans(ex.utt, ex.tokens, spans_to_bio(ex.tokens, truth["slots"])),
                truth["slots"],
            )

    def test_unknown_words_have_distinct_character_inputs(self):
        a = encode_input("ninefriday", "a", self.metadata)
        b = encode_input("fridaynine", "b", self.metadata)
        self.assertEqual(a.input_ids, [1])
        self.assertEqual(b.input_ids, [1])
        self.assertNotEqual(a.char_ids, b.char_ids)

    def test_final_scoring_keeps_unrepresentable_gold_span(self):
        # A contract-only malformed-for-this-tokenizer fixture; never trained on.
        record = dict(RECORDS[0], utt="nineam", annot_utt="[time : nine]am")
        data = EvaluationDataset([record], self.metadata)
        self.assertEqual(len(data), 1)
        self.assertEqual(data.gold[0]["slots"][0]["text"], "nine")

    def test_padding_invariant_predictions(self):
        for chars, pooling in itertools.product((False, True), ("mean_max", "attention")):
            model = self.model(chars=chars, pooling=pooling).eval()
            short = encode_input("nine", "a", self.metadata)
            long = self.dataset.examples[1]
            a, b = collate_inputs([short]), collate_inputs([short, long])
            with torch.inference_mode():
                p = model(a["input_ids"], a["lengths"], a["char_ids"])
                q = model(b["input_ids"], b["lengths"], b["char_ids"])
            torch.testing.assert_close(p[0][0], q[0][0])
            torch.testing.assert_close(p[1][0], q[1][0, :1])

    def test_both_tasks_and_shared_encoder_receive_gradients(self):
        # Intent fixture has one class; add a second numerical output class to
        # exercise a nontrivial intent loss without inventing training records.
        for pooling in ("mean_max", "attention"):
            model = self.model(chars=True, pooling=pooling)
            model.intent_head = torch.nn.Linear(model.intent_head.in_features, 2)
            batch = collate_supervised([self.dataset[0], self.dataset[1]])
            logits = model(batch["input_ids"], batch["lengths"], batch["char_ids"])
            loss, *_ = joint_losses(logits, batch, torch.device("cpu"), model=model)
            loss.backward()
            layers = [model.intent_head, model.slot_head, model.encoder, model.char_conv]
            if model.intent_attention is not None:
                layers.append(model.intent_attention)
            for layer in layers:
                self.assertTrue(
                    any(
                        p.grad is not None and bool(p.grad.abs().sum() > 0)
                        for p in layer.parameters()
                    )
                )
            self.assertTrue(
                all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                )
            )

    def test_attention_normalizes_over_real_tokens_and_ignores_padding(self):
        pool = MaskedAttentionPooling(4, 3)
        encoded = torch.randn(2, 4, 4, requires_grad=True)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        pooled, weights = pool(encoded, mask)
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        self.assertTrue(bool((weights[~mask] == 0).all()))
        changed = encoded.detach().clone()
        changed[~mask] = 10000.0
        torch.testing.assert_close(pool(changed, mask)[0], pooled)
        pooled.square().sum().backward()
        self.assertTrue(bool((encoded.grad[~mask] == 0).all()))
        self.assertTrue(bool(torch.isfinite(encoded.grad).all()))
        self.assertTrue(
            any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in pool.parameters())
        )
        with torch.no_grad():
            for parameter in pool.parameters():
                parameter.zero_()
        uniform, _ = pool(encoded.detach(), mask)
        expected = torch.stack([encoded[0, :2].mean(0), encoded[1, :3].mean(0)])
        torch.testing.assert_close(uniform, expected)

    def test_padding_does_not_change_loss(self):
        model = self.model().eval()
        batch = collate_supervised([self.dataset[0], self.dataset[1]])
        intent, slot = model(batch["input_ids"], batch["lengths"], batch["char_ids"])
        a = joint_losses((intent, slot), batch, torch.device("cpu"))[0]
        expanded = dict(
            batch, slot_labels=torch.cat((batch["slot_labels"], torch.full((2, 3), -100)), 1)
        )
        b = joint_losses(
            (intent, torch.cat((slot, torch.full((2, 3, slot.shape[-1]), 1000.0)), 1)),
            expanded,
            torch.device("cpu"),
        )[0]
        torch.testing.assert_close(a, b)

    def test_frozen_embeddings_still_count(self):
        model = self.model()
        total = parameter_counts(model)["total"]
        model.embedding.weight.requires_grad_(False)
        counts = parameter_counts(model)
        self.assertEqual(counts["total"], total)
        self.assertEqual(counts["trainable"], total - model.embedding.weight.numel())

    def test_word_dropout_preserves_inputs_and_is_disabled_at_evaluation(self):
        model = self.model()
        batch = collate_inputs(self.dataset.examples)
        original = batch["input_ids"].clone()
        model.config.word_dropout = 0.5
        model.train()
        model(batch["input_ids"], batch["lengths"], batch["char_ids"])
        torch.testing.assert_close(batch["input_ids"], original)
        model.eval()
        with torch.inference_mode():
            first = model(batch["input_ids"], batch["lengths"], batch["char_ids"])
            model.config.word_dropout = 0.0
            second = model(batch["input_ids"], batch["lengths"], batch["char_ids"])
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_predictor_reload_and_raw_text_only(self):
        for decoder, pooling in itertools.product(
            ("softmax", "constrained", "crf"), ("mean_max", "attention")
        ):
            model = self.model(chars=True, decoder=decoder, pooling=pooling).eval()
            payload = {
                "model_config": asdict(model.config),
                "model_state": model.state_dict(),
                "metadata": self.metadata,
                "epoch": 0,
                "dev_metrics": {},
                "selection_score": None,
                "data_sha256": "unit-test-only",
            }
            if pooling == "mean_max":
                del payload["model_config"][
                    "intent_pooling"
                ]  # Historical checkpoints lack this field.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                torch.save(payload, path)
                first, second = JointPredictor(path), JointPredictor(path)
                texts = [record["utt"] for record in RECORDS]
                self.assertEqual(first.predict(texts, 1), second.predict(texts, 2))
                with self.assertRaises(TypeError):
                    first.predict(RECORDS)

    def test_exact_training_state_resume(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        generator = torch.Generator().manual_seed(13)
        batch = collate_supervised([self.dataset[0], self.dataset[1]])

        def step(network, optim):
            optim.zero_grad()
            logits = network(batch["input_ids"], batch["lengths"], batch["char_ids"])
            loss = joint_losses(logits, batch, torch.device("cpu"), model=network)[0]
            loss.backward()
            optim.step()
            return float(loss.detach())

        step(model, optimizer)
        scheduler.step(1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_training_state(path, model, optimizer, scheduler, generator)
            expected = step(model, optimizer)
            expected_weights = {key: value.clone() for key, value in model.state_dict().items()}
            restored = self.model()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
            restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer)
            restore_training_state(
                torch.load(path, weights_only=True),
                restored,
                restored_optimizer,
                restored_scheduler,
                torch.Generator(),
            )
            self.assertEqual(expected, step(restored, restored_optimizer))
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)


class TestBIOCRF(unittest.TestCase):
    def setUp(self):
        self.crf = BIOCRF(["O", "B-time", "I-time"]).double()
        self.emissions = torch.tensor(
            [[[0.3, 0.7, 4.0], [0.2, 0.1, 1.3], [0.1, 0.4, 0.8]]],
            dtype=torch.float64,
            requires_grad=True,
        )

    def test_likelihood_and_viterbi_against_exhaustive_paths(self):
        scores, paths = [], []
        for path in itertools.product(range(3), repeat=3):
            if not self.crf.start_allowed[path[0]] or any(
                not self.crf.allowed[a, b] for a, b in zip(path, path[1:])
            ):
                continue
            paths.append(list(path))
            scores.append(sum(self.emissions[0, time, tag] for time, tag in enumerate(path)))
        scores = torch.stack(scores)
        gold = torch.tensor([[1, 2, 0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        expected = torch.logsumexp(scores, 0) - sum(
            self.emissions[0, time, tag] for time, tag in enumerate(gold[0])
        )
        actual = self.crf.neg_log_likelihood(self.emissions, gold, mask)[0]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(self.crf.decode(self.emissions, mask), [paths[int(scores.argmax())]])
        actual.backward()
        self.assertTrue(bool(torch.isfinite(self.emissions.grad).all()))

    def test_decoding_is_legal_and_ignores_padding(self):
        mask = torch.tensor([[True, True, False]])
        first = self.crf.decode(self.emissions, mask)
        changed = self.emissions.detach().clone()
        changed[:, 2] = torch.tensor([100.0, -100.0, 500.0])
        self.assertEqual(first, self.crf.decode(changed, mask))
        self.assertNotEqual(first[0][0], 2)

    def test_illegal_gold_is_rejected(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.crf.neg_log_likelihood(self.emissions, torch.tensor([[2, 2, 0]]), mask)
        with self.assertRaises(ValueError):
            self.crf.neg_log_likelihood(self.emissions, torch.tensor([[0, 2, 0]]), mask)


if __name__ == "__main__":
    unittest.main()
