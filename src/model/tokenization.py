"""Text-only tokenization with reversible raw-text character coordinates."""

import re
from collections import Counter
from dataclasses import dataclass

PAD, UNK = "<pad>", "<unk>"


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(utt: str) -> list[Token]:
    if not isinstance(utt, str):
        raise TypeError("The predictor accepts raw utt strings only")
    tokens = [Token(m.group(), *m.span()) for m in re.finditer(r"\S+", utt)]
    if not tokens:
        raise ValueError("Utterance must contain a non-whitespace token")
    return tokens


def build_vocab(train_records: list[dict], min_frequency: int = 1) -> dict[str, int]:
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")
    counts = Counter()
    for record in train_records:
        if record["partition"] != "train":
            raise ValueError("Vocabulary may only be fitted on Train")
        counts.update(token.text.lower() for token in tokenize(record["utt"]))
    vocab = {PAD: 0, UNK: 1}
    for word in sorted(counts, key=lambda word: (-counts[word], word)):
        if counts[word] >= min_frequency and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def build_bio_maps(slot_types: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    names = ["O"]
    for kind in sorted(set(slot_types)):
        names.extend((f"B-{kind}", f"I-{kind}"))
    return dict(zip(names, range(len(names)))), dict(enumerate(names))


def spans_to_bio(tokens: list[Token], slots: list[dict]) -> list[str]:
    tags = ["O"] * len(tokens)
    starts = {token.start: i for i, token in enumerate(tokens)}
    ends = {token.end: i for i, token in enumerate(tokens)}
    for slot in slots:
        if slot["start"] not in starts or slot["end"] not in ends:
            raise ValueError(f"Gold span cannot align to text-only tokens: {slot}")
        first, last = starts[slot["start"]], ends[slot["end"]]
        if first > last or any(tag != "O" for tag in tags[first : last + 1]):
            raise ValueError("Overlapping or reversed gold slot")
        tags[first] = f"B-{slot['type']}"
        for i in range(first + 1, last + 1):
            tags[i] = f"I-{slot['type']}"
    return tags


def bio_to_spans(utt: str, tokens: list[Token], tags: list[str]) -> list[dict]:
    """Decode exact character spans; orphan/wrong-type I starts a new span.

    This fixed repair policy is applied identically to every experiment.
    B always starts a new span, including adjacent spans of the same type.
    """
    if len(tokens) != len(tags):
        raise ValueError("One BIO tag is required for every input token")
    spans = []
    active_type, active_start, active_end = None, None, None

    def flush():
        if active_type is not None:
            spans.append(
                {
                    "type": active_type,
                    "start": active_start,
                    "end": active_end,
                    "text": utt[active_start:active_end],
                }
            )

    for token, tag in zip(tokens, tags):
        if not (0 <= token.start < token.end <= len(utt)):
            raise ValueError("Token is outside the raw utterance")
        if utt[token.start : token.end] != token.text:
            raise ValueError("Token does not match raw utterance")
        if tag == "O":
            flush()
            active_type = None
            continue
        prefix, separator, kind = tag.partition("-")
        if prefix not in ("B", "I") or not separator or not kind:
            raise ValueError(f"Invalid BIO tag: {tag!r}")
        if prefix == "B" or active_type != kind:
            flush()
            active_type, active_start = kind, token.start
        active_end = token.end
    flush()
    return spans
