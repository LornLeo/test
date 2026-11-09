"""Train-only MASSIVE label statistics and ID mappings."""

from collections import Counter
from collections.abc import Iterable

from src.data.annotations import extract_slots


def count_train_labels(
    train_records: Iterable[dict],
) -> tuple[Counter, Counter]:
    """Count intents per utterance and slot types per span."""
    intent_counts = Counter()
    slot_counts = Counter()

    for index, record in enumerate(train_records):
        if record["partition"] != "train":
            raise ValueError(
                f"Record {index} is not in Train"
            )

        intent = record["intent"]
        if not isinstance(intent, str) or not intent:
            raise ValueError(
                f"Record {index} has invalid intent: {intent!r}"
            )

        slots = extract_slots(
            record["utt"],
            record["annot_utt"],
        )

        intent_counts[intent] += 1
        slot_counts.update(slot["type"] for slot in slots)

    return intent_counts, slot_counts


def build_label_maps(
    labels: Iterable[str],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build reproducible, alphabetically sorted label IDs."""

    if isinstance(labels, (str, bytes)):
        raise TypeError(
            "labels must be an iterable of strings, "
            "not a single string"
        )

    unique_labels = set()

    for label in labels:
        if not isinstance(label, str):
            raise TypeError(
                f"Label must be a string, got "
                f"{type(label).__name__}"
            )

        if not label.strip():
            raise ValueError("Label cannot be empty or blank")

        unique_labels.add(label)

    names = sorted(unique_labels)

    label_to_id = {
        name: index
        for index, name in enumerate(names)
    }

    id_to_label = {
        index: name
        for name, index in label_to_id.items()
    }

    return label_to_id, id_to_label
