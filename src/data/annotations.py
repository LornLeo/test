"""Parse MASSIVE 1.0 annot_utt into candidate Python-character spans.

The output span convention [start, end) is provisional until agreed with B.
No normalization is performed: reconstructed text must equal the raw utt.
"""

import re

# MASSIVE's documented annotation form: [slot_type : slot_text]
_SLOT = re.compile(
    r"\[(?P<type>[A-Za-z_][A-Za-z0-9_]*) : (?P<text>[^\[\]]+)\]"
)


def extract_slots(utt: str, annot_utt: str) -> list[dict]:
    """Return slots with character offsets into the original utt.

    Raises ValueError if the annotated text cannot reconstruct utt exactly.
    Offsets use Python string indices.
    """
    if not isinstance(utt, str) or not isinstance(annot_utt, str):
        raise TypeError("utt and annot_utt must be strings")

    parts = []
    slots = []
    cursor = 0
    position = 0

    for match in _SLOT.finditer(annot_utt):
        plain_text = annot_utt[cursor:match.start()]
        parts.append(plain_text)
        position += len(plain_text)

        slot_text = match.group("text")
        start = position
        end = start + len(slot_text)

        slots.append({
            "type": match.group("type"),
            "start": start,
            "end": end,
            "text": slot_text,
        })

        parts.append(slot_text)
        position = end
        cursor = match.end()

    parts.append(annot_utt[cursor:])
    reconstructed = "".join(parts)

    if reconstructed != utt:
        raise ValueError(
            "Annotation cannot be reconstructed exactly as utt: "
            f"utt={utt!r}, reconstructed={reconstructed!r}"
        )

    # Validate offsets without searching by text.
    for slot in slots:
        if utt[slot["start"]:slot["end"]] != slot["text"]:
            raise ValueError(f"Slot offset mismatch: {slot!r}")

    return slots
