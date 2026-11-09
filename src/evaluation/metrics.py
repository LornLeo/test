"""Evaluate MASSIVE intent and exact type+span slot predictions.

Input: aligned sequences of prediction and gold records:
{"intent": str, "slots": [{"type": str, "start": int, "end": int}, ...]}

An optional "id" on both records is used to verify sample order.
Slots use provisional [start, end) offsets into raw utt.
"""

from collections import Counter
from collections.abc import Sequence


def _slot_counter(record: dict, index: int, side: str) -> Counter:
    slots = record.get("slots")
    if not isinstance(slots, list):
        raise ValueError(f"{side}[{index}].slots must be a list")

    counts = Counter()
    for j, slot in enumerate(slots):
        if not isinstance(slot, dict):
            raise ValueError(f"{side}[{index}].slots[{j}] must be a dict")

        try:
            kind, start, end = (
                slot["type"], slot["start"], slot["end"]
            )
        except KeyError as exc:
            raise ValueError(
                f"{side}[{index}].slots[{j}] "
                f"missing {exc.args[0]!r}"
            ) from exc

        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(
                f"{side}[{index}].slots[{j}] has invalid type"
            )

        if (
            type(start) is not int
            or type(end) is not int
            or not (0 <= start < end)
        ):
            raise ValueError(
                f"{side}[{index}].slots[{j}] has invalid span"
            )

        counts[(kind, start, end)] += 1

    return counts


def evaluate(predictions: Sequence[dict], gold: Sequence[dict]) -> dict:
    """Return unrounded percentage metrics and slot TP/FP/FN.

    Slots are matched within each sample.
    Duplicate predictions count separately.
    Empty evaluation datasets are rejected.
    """

    if len(predictions) != len(gold):
        raise ValueError(
            f"Sample count mismatch: {len(predictions)} predictions "
            f"vs {len(gold)} gold records"
        )

    if not gold:
        raise ValueError("Cannot evaluate an empty dataset")

    correct_intents = tp = fp = fn = 0

    id_mode = None
    seen_pred_ids = set()
    seen_gold_ids = set()

    for i, (pred, truth) in enumerate(zip(predictions, gold)):
        if not isinstance(pred, dict) or not isinstance(truth, dict):
            raise ValueError(
                f"Sample {i} must contain two dictionaries"
            )

        pred_has_id = "id" in pred
        gold_has_id = "id" in truth

        if pred_has_id != gold_has_id:
            raise ValueError(
                f"Sample {i}: ID provided on only one side"
            )

        if id_mode is None:
            id_mode = pred_has_id
        elif pred_has_id != id_mode:
            raise ValueError(
                f"Sample {i}: inconsistent ID presence"
            )

        if pred_has_id:
            for side, record, seen in (
                ("predictions", pred, seen_pred_ids),
                ("gold", truth, seen_gold_ids),
            ):
                record_id = record["id"]

                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(
                        f"{side}[{i}].id must be a nonempty string"
                    )

                if record_id in seen:
                    raise ValueError(
                        f"{side}[{i}]: duplicate ID {record_id!r}"
                    )

                seen.add(record_id)

            if pred["id"] != truth["id"]:
                raise ValueError(
                    f"Sample order mismatch at index {i}: "
                    f"prediction id={pred['id']!r}, "
                    f"gold id={truth['id']!r}"
                )

        for side, record in (
            ("predictions", pred),
            ("gold", truth),
        ):
            intent = record.get("intent")
            if not isinstance(intent, str) or not intent.strip():
                raise ValueError(
                    f"{side}[{i}].intent must be a nonempty string"
                )

        correct_intents += pred["intent"] == truth["intent"]

        predicted_slots = _slot_counter(pred, i, "predictions")
        gold_slots = _slot_counter(truth, i, "gold")

        duplicate_gold = [
            key for key, count in gold_slots.items()
            if count > 1
        ]

        if duplicate_gold:
            raise ValueError(
                f"Sample {i}: duplicate gold slot "
                f"{duplicate_gold[0]!r}"
            )

        # Counter intersection preserves duplicate counts.
        matched = sum(
            (predicted_slots & gold_slots).values()
        )

        tp += matched
        fp += sum(predicted_slots.values()) - matched
        fn += sum(gold_slots.values()) - matched

    denominator = 2 * tp + fp + fn

    return {
        "intent_accuracy_pct": (
            100 * correct_intents / len(gold)
        ),
        "slot_micro_f1_pct": (
            100 * 2 * tp / denominator
            if denominator else 0.0
        ),
        "slot_tp": tp,
        "slot_fp": fp,
        "slot_fn": fn,
    }
