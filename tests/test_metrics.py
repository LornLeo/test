import unittest

from src.evaluation.metrics import evaluate


def sample(intent="alarm_set", slots=None, sample_id=None):
    record = {
        "intent": intent,
        "slots": [] if slots is None else slots,
    }
    if sample_id is not None:
        record["id"] = sample_id
    return record


def slot(kind="time", start=4, end=9):
    return {
        "type": kind,
        "start": start,
        "end": end,
    }


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        gold = [
            sample(slots=[slot()], sample_id="a"),
            sample("weather_query", sample_id="b"),
        ]

        self.assertEqual(evaluate(gold, gold), {
            "intent_accuracy_pct": 100.0,
            "slot_micro_f1_pct": 100.0,
            "slot_tp": 1,
            "slot_fp": 0,
            "slot_fn": 0,
        })

    def test_intent_accuracy(self):
        gold = [
            sample("alarm_set"),
            sample("weather_query"),
        ]
        predicted = [
            sample("alarm_set"),
            sample("alarm_set"),
        ]

        self.assertEqual(
            evaluate(predicted, gold)["intent_accuracy_pct"],
            50.0,
        )

    def test_wrong_type_and_span(self):
        gold = [
            sample(slots=[slot()]),
            sample(slots=[slot()]),
        ]
        predicted = [
            sample(slots=[slot(kind="date")]),
            sample(slots=[slot(start=5, end=9)]),
        ]

        result = evaluate(predicted, gold)
        self.assertEqual(
            (
                result["slot_tp"],
                result["slot_fp"],
                result["slot_fn"],
            ),
            (0, 2, 2),
        )
        self.assertEqual(result["slot_micro_f1_pct"], 0.0)

    def test_repeated_predictions_count_extra_fp(self):
        gold = [sample(slots=[slot()])]
        predicted = [sample(slots=[slot(), slot()])]

        result = evaluate(predicted, gold)

        self.assertEqual(
            (
                result["slot_tp"],
                result["slot_fp"],
                result["slot_fn"],
            ),
            (1, 1, 0),
        )
        self.assertAlmostEqual(
            result["slot_micro_f1_pct"],
            200 / 3,
        )

    def test_micro_aggregation_and_missing_slot(self):
        gold = [
            sample(slots=[slot(), slot("date", 15, 21)]),
            sample(),
        ]
        predicted = [
            sample(slots=[slot()]),
            sample(slots=[slot()]),
        ]

        result = evaluate(predicted, gold)

        self.assertEqual(
            (
                result["slot_tp"],
                result["slot_fp"],
                result["slot_fn"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(
            result["slot_micro_f1_pct"],
            50.0,
        )

    def test_no_slot_in_nonempty_dataset(self):
        result = evaluate([sample()], [sample()])

        self.assertEqual(
            result["slot_micro_f1_pct"], 0.0
        )
        self.assertEqual(
            result["intent_accuracy_pct"], 100.0
        )

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([], [])

    def test_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(
                [sample()],
                [sample(), sample()],
            )

    def test_no_cross_sample_slot_matching(self):
        gold = [
            sample(slots=[slot()]),
            sample(),
        ]
        predicted = [
            sample(),
            sample(slots=[slot()]),
        ]

        result = evaluate(predicted, gold)

        self.assertEqual(
            (
                result["slot_tp"],
                result["slot_fp"],
                result["slot_fn"],
            ),
            (0, 1, 1),
        )

    def test_id_order_mismatch_rejected(self):
        gold = [
            sample(sample_id="a"),
            sample(sample_id="b"),
        ]
        predicted = [
            sample(sample_id="b"),
            sample(sample_id="a"),
        ]

        with self.assertRaises(ValueError):
            evaluate(predicted, gold)

    def test_invalid_span_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(
                [sample(slots=[slot(start=5, end=4)])],
                [sample()],
            )

    def test_one_sided_id_rejected(self):
        gold = [sample(sample_id="a")]
        predicted = [sample()]

        with self.assertRaises(ValueError):
            evaluate(predicted, gold)

    def test_partial_ids_rejected(self):
        gold = [
            sample(sample_id="a"),
            sample(),
        ]
        predicted = [
            sample(sample_id="a"),
            sample(),
        ]

        with self.assertRaises(ValueError):
            evaluate(predicted, gold)

    def test_duplicate_ids_rejected(self):
        gold = [
            sample(sample_id="a"),
            sample(sample_id="a"),
        ]
        predicted = [
            sample(sample_id="a"),
            sample(sample_id="a"),
        ]

        with self.assertRaises(ValueError):
            evaluate(predicted, gold)

    def test_duplicate_gold_slots_rejected(self):
        gold = [
            sample(slots=[slot(), slot()])
        ]
        predicted = [
            sample(slots=[slot()])
        ]

        with self.assertRaises(ValueError):
            evaluate(predicted, gold)


if __name__ == "__main__":
    unittest.main()
