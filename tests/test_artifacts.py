"""Audit fingerprints must survive reserialization and detect model changes."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.model.artifacts import model_state_sha256


class TestModelFingerprint(unittest.TestCase):
    def test_archive_path_and_dictionary_order_do_not_change_fingerprint(self):
        state = {"weight": torch.arange(6.0).reshape(2, 3), "mask": torch.tensor([True, False])}
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first.pt", Path(temporary) / "second.pt"
            torch.save(state, first)
            torch.save(dict(reversed(list(state.items()))), second)
            a, b = (torch.load(p, weights_only=True) for p in (first, second))
            self.assertEqual(model_state_sha256(a), model_state_sha256(b))

    def test_values_shapes_and_dtypes_are_part_of_fingerprint(self):
        value = torch.arange(6.0)
        original = model_state_sha256({"weight": value})
        for changed in (value + 1, value.reshape(2, 3), value.double()):
            self.assertNotEqual(original, model_state_sha256({"weight": changed}))
