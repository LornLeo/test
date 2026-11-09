import unittest

from src.data.labels import (
    build_label_maps,
    count_train_labels,
)


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "partition": "train",
                "intent": "alarm_set",
                "utt": "wake me at seven on friday",
                "annot_utt": (
                    "wake me at [time : seven] "
                    "on [date : friday]"
                ),
            },
            {
                "partition": "train",
                "intent": "weather_query",
                "utt": "what is the weather",
                "annot_utt": "what is the weather",
            },
            {
                "partition": "train",
                "intent": "alarm_set",
                "utt": "seven then eight",
                "annot_utt": (
                    "[time : seven] then [time : eight]"
                ),
            },
        ]

    def test_counts(self):
        intents, slots = count_train_labels(self.records)

        self.assertEqual(
            dict(intents),
            {"alarm_set": 2, "weather_query": 1},
        )

        self.assertEqual(
            dict(slots),
            {"time": 3, "date": 1},
        )

    def test_sorted_mapping_and_round_trip(self):
        label_to_id, id_to_label = build_label_maps(
            ["time", "date", "time"]
        )

        self.assertEqual(
            label_to_id,
            {"date": 0, "time": 1},
        )

        self.assertEqual(
            id_to_label,
            {0: "date", 1: "time"},
        )

        for label, index in label_to_id.items():
            self.assertEqual(id_to_label[index], label)

    def test_mapping_is_order_independent(self):
        first = build_label_maps(["b", "a", "c"])
        second = build_label_maps(["c", "b", "a"])

        self.assertEqual(first, second)

    def test_reject_nontrain(self):
        wrong = dict(
            self.records[0],
            partition="dev",
        )

        with self.assertRaises(ValueError):
            count_train_labels([wrong])

    def test_reject_single_string(self):
        with self.assertRaises(TypeError):
            build_label_maps("alarm_set")

    def test_reject_invalid_labels(self):
        for labels in ([""], ["  "]):
            with self.assertRaises(ValueError):
                build_label_maps(labels)

        for labels in ([None], ["time", 123]):
            with self.assertRaises(TypeError):
                build_label_maps(labels)


if __name__ == "__main__":
    unittest.main()
