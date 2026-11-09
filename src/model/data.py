"""A/B adapter: Train-only metadata, BIO supervision, and dynamic padding."""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.data.annotations import extract_slots
from src.data.labels import build_label_maps, count_train_labels
from src.model.tokenization import (
    UNK,
    Token,
    bio_to_spans,
    build_bio_maps,
    build_vocab,
    spans_to_bio,
    tokenize,
)

IGNORE_INDEX = -100


def fit_metadata(train_records: list[dict], min_frequency: int = 1) -> dict:
    intent_counts, slot_counts = count_train_labels(train_records)
    intent_to_id, _ = build_label_maps(intent_counts)
    slot_to_id, _ = build_label_maps(slot_counts)
    bio_to_id, _ = build_bio_maps(list(slot_to_id))
    characters = sorted({char for record in train_records for char in record["utt"]})
    char_vocab = {"<pad>": 0, "<unk>": 1}
    char_vocab.update({char: i + 2 for i, char in enumerate(characters)})
    return {
        "vocab": build_vocab(train_records, min_frequency),
        "char_vocab": char_vocab,
        "intent_to_id": intent_to_id,
        "slot_to_id": slot_to_id,
        "bio_to_id": bio_to_id,
        "tokenizer": "whitespace/raw-codepoints/v1",
    }


@dataclass
class EncodedInput:
    sample_id: str
    utt: str
    tokens: list[Token]
    input_ids: list[int]
    char_ids: list[list[int]]


def encode_input(utt: str, sample_id: str, metadata: dict) -> EncodedInput:
    tokens = tokenize(utt)
    vocab = metadata["vocab"]
    char_vocab = metadata.get("char_vocab", {"<pad>": 0, "<unk>": 1})
    return EncodedInput(
        sample_id,
        utt,
        tokens,
        [vocab.get(t.text.lower(), vocab[UNK]) for t in tokens],
        [[char_vocab.get(char, 1) for char in t.text] for t in tokens],
    )


def collate_inputs(examples: list[EncodedInput]) -> dict:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(ex.tokens) for ex in examples], dtype=torch.long)
    ids = torch.zeros((len(examples), int(lengths.max())), dtype=torch.long)
    max_chars = max(len(chars) for ex in examples for chars in ex.char_ids)
    char_ids = torch.zeros((*ids.shape, max_chars), dtype=torch.long)
    for row, ex in enumerate(examples):
        ids[row, : len(ex.input_ids)] = torch.tensor(ex.input_ids)
        for column, values in enumerate(ex.char_ids):
            char_ids[row, column, : len(values)] = torch.tensor(values)
    return {"input_ids": ids, "lengths": lengths, "char_ids": char_ids, "examples": examples}


class JointDataset(Dataset):
    def __init__(self, records: list[dict], metadata: dict):
        self.examples, self.targets, self.gold = [], [], []
        for record in records:
            ex = encode_input(record["utt"], record["id"], metadata)
            slots = extract_slots(record["utt"], record["annot_utt"])
            tags = spans_to_bio(ex.tokens, slots)
            decoded = bio_to_spans(ex.utt, ex.tokens, tags)
            if decoded != slots:
                raise ValueError(f"Gold BIO round-trip failed: {record['id']}")
            self.examples.append(ex)
            self.targets.append(
                (
                    metadata["intent_to_id"][record["intent"]],
                    [metadata["bio_to_id"][tag] for tag in tags],
                )
            )
            self.gold.append({"id": record["id"], "intent": record["intent"], "slots": slots})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index], self.targets[index]


class EvaluationDataset:
    """Raw-span gold for final scoring, without requiring BIO-representable gold.

    Even a gold span that this tokenizer cannot express must remain in scoring
    as a possible false negative, rather than cause an example to be dropped.
    """

    def __init__(self, records: list[dict], metadata: dict):
        self.examples = [encode_input(record["utt"], record["id"], metadata) for record in records]
        self.gold = [
            {
                "id": record["id"],
                "intent": record["intent"],
                "slots": extract_slots(record["utt"], record["annot_utt"]),
            }
            for record in records
        ]

    def __len__(self):
        return len(self.examples)


def collate_supervised(items: list) -> dict:
    batch = collate_inputs([item[0] for item in items])
    intents, tags = zip(*(item[1] for item in items))
    labels = torch.full_like(batch["input_ids"], IGNORE_INDEX)
    for row, values in enumerate(tags):
        labels[row, : len(values)] = torch.tensor(values)
    batch.update(intent_labels=torch.tensor(intents), slot_labels=labels)
    return batch
