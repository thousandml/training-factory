import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from residual_names import resolve_residual

class OptionalMtpTests(unittest.TestCase):
    def test_absent_mtp_residual_ignored(self):
        self.assertIsNone(resolve_residual('model.mtp.fc.weight', {'model.visual.fc.weight': object()}, text_only=False))

    def test_present_mtp_residual_strict(self):
        with self.assertRaises(ValueError):
            resolve_residual('model.mtp.fc.weight', {'model.mtp.other.weight': object()}, text_only=False)

    def test_backbone_missing_still_errors(self):
        with self.assertRaises(ValueError):
            resolve_residual('model.layers.1.fc.weight', {'model.layers.0.fc.weight': object()})
