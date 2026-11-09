
"""A-side integration checks for MASSIVE 1.0.

Build label mappings from Train only.
Validate and self-score Train and Dev separately.
Check Test data integrity without scoring it.
"""

from pathlib import Path

from src.data.loader import load_massive
from src.data.annotations import extract_slots
from src.data.labels import count_train_labels, build_label_maps
from src.evaluation.metrics import evaluate


def check_ids(records, split):
    """Validate optional sample IDs within one partition."""
    present = ["id" in record for record in records]

    if any(present) and not all(present):
        raise ValueError(
            f"{split}: sample IDs are only partially present"
        )

    has_ids = all(present)

    if has_ids:
        seen = set()

        for index, record in enumerate(records):
            sample_id = record["id"]

            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"{split}[{index}]: invalid sample ID "
                    f"{sample_id!r}"
                )

            if sample_id in seen:
                raise ValueError(
                    f"{split}[{index}]: duplicate sample ID "
                    f"{sample_id!r}"
                )

            seen.add(sample_id)

    return has_ids


def check_label(label, to_id, from_id, context):
    """Verify Train mapping coverage and ID round-trip."""
    if label not in to_id:
        raise ValueError(
            f"{context}: unknown Train label {label!r}"
        )

    if from_id[to_id[label]] != label:
        raise ValueError(
            f"{context}: label ID round-trip failed "
            f"for {label!r}"
        )


def make_gold(records, split, intent_maps, slot_maps):
    """Convert records to evaluator Gold format."""
    has_ids = check_ids(records, split)

    intent_to_id, id_to_intent = intent_maps
    slot_to_id, id_to_slot = slot_maps

    gold = []
    total_slots = 0

    for index, record in enumerate(records):
        context = (
            f"{split}[{index}] id={record.get('id')!r}"
        )

        if record["partition"] != split:
            raise ValueError(
                f"{context}: wrong partition"
            )

        check_label(
            record["intent"],
            intent_to_id,
            id_to_intent,
            context,
        )

        try:
            parsed_slots = extract_slots(
                record["utt"],
                record["annot_utt"],
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{context}: annotation error: {exc}"
            ) from exc

        clean_slots = []

        for slot in parsed_slots:
            check_label(
                slot["type"],
                slot_to_id,
                id_to_slot,
                context,
            )

            if (
                record["utt"][slot["start"]:slot["end"]]
                != slot["text"]
            ):
                raise ValueError(
                    f"{context}: slot slice mismatch"
                )

            clean_slots.append({
                "type": slot["type"],
                "start": slot["start"],
                "end": slot["end"],
            })

        gold_record = {
            "intent": record["intent"],
            "slots": clean_slots,
        }

        if has_ids:
            gold_record["id"] = record["id"]

        gold.append(gold_record)
        total_slots += len(clean_slots)

    if len(gold) != len(records):
        raise RuntimeError(
            f"{split}: record count changed"
        )

    return gold, total_slots


def check_test_integrity(records):
    """Check Test reconstruction without scoring or label fitting."""
    check_ids(records, "test")

    for index, record in enumerate(records):
        context = f"test[{index}] id={record.get('id')!r}"

        if record["partition"] != "test":
            raise ValueError(
                f"{context}: wrong partition"
            )

        try:
            extract_slots(
                record["utt"],
                record["annot_utt"],
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{context}: annotation error: {exc}"
            ) from exc

    print(
        f"TEST: {len(records)} samples, "
        "reconstruction check passed; not scored"
    )


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "data/1.0/data/en-US.jsonl"

    dataset = load_massive(path)

    # Build all mappings exclusively from Train.
    intent_counts, slot_counts = count_train_labels(
        dataset["train"]
    )

    intent_maps = build_label_maps(intent_counts.keys())
    slot_maps = build_label_maps(slot_counts.keys())

    print(
        f"Train-only label maps: "
        f"{len(intent_maps[0])} intents, "
        f"{len(slot_maps[0])} slot types"
    )

    # Validate and self-score Train and Dev independently.
    for split in ("train", "dev"):
        records = dataset[split]

        gold, slot_count = make_gold(
            records,
            split,
            intent_maps,
            slot_maps,
        )

        result = evaluate(gold, gold)

        expected_f1 = 100.0 if slot_count else 0.0

        if not (
            result["intent_accuracy_pct"] == 100.0
            and result["slot_micro_f1_pct"] == expected_f1
            and result["slot_tp"] == slot_count
            and result["slot_fp"] == 0
            and result["slot_fn"] == 0
        ):
            raise RuntimeError(
                f"{split}: Gold self-score failed: {result}"
            )

        print(
            f"{split.upper()}: "
            f"samples={len(records)}, "
            f"gold slots={slot_count}"
        )

        print(
            f"  Train-map round trips: "
            f"intents={len(records)}/{len(records)}, "
            f"slots={slot_count}/{slot_count}"
        )

        print(
            f"  Self-score: "
            f"intent={result['intent_accuracy_pct']:.2f}%, "
            f"slot F1={result['slot_micro_f1_pct']:.2f}%; "
            f"TP={result['slot_tp']}, "
            f"FP={result['slot_fp']}, "
            f"FN={result['slot_fn']}"
        )

    # Test: integrity only. No scoring or mapping updates.
    check_test_integrity(dataset["test"])


if __name__ == "__main__":
    main()
