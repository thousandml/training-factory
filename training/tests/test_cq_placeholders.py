import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from residual_placeholders import is_cq_placeholder

class TestCQPlaceholders(unittest.TestCase):
    def test_cq_backed_empty(self):
        self.assertTrue(is_cq_placeholder('model.language_model.layers.0.linear_attn.in_proj_qkv.weight', [0], 'model.layers.0.linear_attn.in_proj_qkv.weight', {'model.layers.0.linear_attn.in_proj_qkv'}))
    def test_nonempty(self):
        self.assertFalse(is_cq_placeholder('x.weight', [3, 4], 'x.weight', set()))
    def test_empty_uncovered(self):
        with self.assertRaises(ValueError):
            is_cq_placeholder('x.weight', [0], 'x.weight', set())
    def test_empty_nonweight(self):
        with self.assertRaises(ValueError):
            is_cq_placeholder('x.bias', [0], 'x.bias', {'x'})

if __name__ == '__main__': unittest.main()
