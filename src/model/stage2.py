"""Stage 2 (Role D) addition that leaves the Stage 1 pipeline unchanged by default.

Role C's proposal C1: class-weighted intent cross-entropy, with weights computed
from the official Train labels only and applied to the training loss only.
"""

import torch

INTENT_WEIGHT_POWER = 0.5
INTENT_WEIGHT_CLIP = (0.25, 4.0)


def intent_class_weights(train_records: list[dict], intent_to_id: dict) -> torch.Tensor:
    """raw_c = (median_count / count_c) ** 0.5, clipped to [0.25, 4.0], then rescaled so
    the average weight over Train examples is 1 (the intent loss keeps its scale
    relative to the slot loss). Declared in docs/D_STAGE2_PLAN.md before training.
    """
    if any(r["partition"] != "train" for r in train_records):
        raise ValueError("Intent class weights may only be computed from Train")
    counts = torch.zeros(len(intent_to_id))
    for record in train_records:
        counts[intent_to_id[record["intent"]]] += 1
    if (counts == 0).any():
        raise ValueError("Every intent must occur in Train")
    median = counts.median()
    raw = (median / counts).pow(INTENT_WEIGHT_POWER).clamp(*INTENT_WEIGHT_CLIP)
    return raw * counts.sum() / (raw * counts).sum()
