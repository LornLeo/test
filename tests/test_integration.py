import unittest

from scripts.check_a_integration import (
    make_gold,
    check_ids,
    check_test_integrity,
)
from src.data.labels import build_label_maps
from src.evaluation.metrics import evaluate


def rec(split, idx, utt, annot, intent):
    return {
        "partition": split,
        "id": idx,
        "utt": utt,
        "annot_utt": annot,
        "intent": intent,
    }


class TestAIntegration(unittest.TestCase):
    def setUp(self):
        self.intent_maps = build_label_maps([
            "alarm_set",
            "weather_query",
        ])

        self.slot_maps = build_label_maps(["time"])

    def test_dev_roundtrip_and_selfscore(self):
        data = [
            rec(
                "dev",
                "b",
                "set alarm for seven",
                "set alarm for [time : seven]",
                "alarm_set",
            ),
            rec(
                "dev",
                "a",
                "what is the weather",
                "what is the weather",
                "weather_query",
            ),
        ]

        gold, count = make_gold(
            data,
            "dev",
            self.intent_maps,
            self.slot_maps,
        )

        # Sample count and original order must be preserved.
        self.assertEqual(len(gold), 2)
        self.assertEqual(
            [record["id"] for record in gold],
            ["b", "a"],
        )
        self.assertEqual(
            [record["intent"] for record in gold],
            ["alarm_set", "weather_query"],
        )

        # Verify slot conversion and total count.
        self.assertEqual(count, 1)
        self.assertEqual(gold[0]["slots"], [
            {"type": "time", "start": 14, "end": 19}
        ])
        self.assertEqual(gold[1]["slots"], [])

        # Verify Gold self-scoring.
        result = evaluate(gold, gold)

        self.assertEqual(
            result["intent_accuracy_pct"], 100.0
        )
        self.assertEqual(
            result["slot_micro_f1_pct"], 100.0
        )
        self.assertEqual(
            (
                result["slot_tp"],
                result["slot_fp"],
                result["slot_fn"],
            ),
            (1, 0, 0),
        )

    def test_unknown_dev_intent(self):
        data = [
            rec("dev", "1", "hello", "hello", "unknown")
        ]

        with self.assertRaisesRegex(
            ValueError, "unknown Train label"
        ):
            make_gold(
                data, "dev",
                self.intent_maps, self.slot_maps,
            )

    def test_unknown_dev_slot(self):
        data = [
            rec(
                "dev", "1",
                "hi sun",
                "hi [date : sun]",
                "alarm_set",
            )
        ]

        with self.assertRaisesRegex(
            ValueError, "unknown Train label"
        ):
            make_gold(
                data, "dev",
                self.intent_maps, self.slot_maps,
            )

    def test_partial_ids(self):
        records = [
            rec("train", "1", "hi", "hi", "alarm_set"),
            rec("train", "2", "hey", "hey", "alarm_set"),
        ]

        del records[1]["id"]

        with self.assertRaises(ValueError):
            check_ids(records, "train")

    def test_test_integrity_valid_sample(self):
        data = [
            rec(
                "test", "3",
                "wake now",
                "wake [time : now]",
                "alarm_set",
            )
        ]

        check_test_integrity(data)


if __name__ == "__main__":
    unittest.main()
