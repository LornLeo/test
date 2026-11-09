import unittest

from src.data.annotations import extract_slots


class TestExtractSlots(unittest.TestCase):
    def test_provided_example(self):
        utt = "set an alarm for two hours from now"
        annotated = "set an alarm for [time : two hours from now]"

        self.assertEqual(extract_slots(utt, annotated), [
            {
                "type": "time",
                "start": 17,
                "end": 35,
                "text": "two hours from now",
            }
        ])

    def test_no_slot(self):
        self.assertEqual(
            extract_slots("hello there", "hello there"),
            [],
        )

    def test_multiple_slots(self):
        utt = "wake me at seven on friday"
        annotated = "wake me at [time : seven] on [date : friday]"

        self.assertEqual(extract_slots(utt, annotated), [
            {
                "type": "time",
                "start": 11,
                "end": 16,
                "text": "seven",
            },
            {
                "type": "date",
                "start": 20,
                "end": 26,
                "text": "friday",
            },
        ])

    def test_same_text_twice(self):
        utt = "remind me at seven and at seven"
        annotated = (
            "remind me at [time : seven] "
            "and at [time : seven]"
        )

        slots = extract_slots(utt, annotated)

        self.assertEqual(
            [slot["start"] for slot in slots],
            [13, 26],
        )
        self.assertEqual(
            [slot["text"] for slot in slots],
            ["seven", "seven"],
        )

    def test_only_second_repeated_mention_is_annotated(self):
        utt = "seven then seven"
        annotated = "seven then [time : seven]"

        self.assertEqual(extract_slots(utt, annotated), [
            {
                "type": "time",
                "start": 11,
                "end": 16,
                "text": "seven",
            }
        ])

    def test_whitespace_is_not_normalized(self):
        with self.assertRaises(ValueError):
            extract_slots(
                "hello  world",
                "hello [object : world]",
            )

    def test_mismatch_is_not_silently_fixed(self):
        with self.assertRaises(ValueError):
            extract_slots(
                "set alarm now",
                "set alarm [time : later]",
            )

    def test_unicode_offsets(self):
        utt = "\u4f60\u597d\U0001f642 \u660e\u5929\u63d0\u9192\u6211"
        annotated = "\u4f60\u597d\U0001f642 [date : \u660e\u5929]\u63d0\u9192\u6211"

        self.assertEqual(extract_slots(utt, annotated), [
            {
                "type": "date",
                "start": 4,
                "end": 6,
                "text": "\u660e\u5929",
            }
        ])

        self.assertEqual(utt[4:6], "\u660e\u5929")


if __name__ == "__main__":
    unittest.main()
