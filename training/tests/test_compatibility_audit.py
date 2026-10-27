import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compatibility_audit import classify_record

class AuditTests(unittest.TestCase):
    def test_vision_text_only(self):
        rec = {'meta': {'matrix_shape': [1152, 4304]}}
        self.assertEqual(classify_record('model.visual.blocks.0.mlp.linear_fc2', rec, text_only=True)[0], 'auxiliary_skippable')
        self.assertEqual(classify_record('model.visual.blocks.0.mlp.linear_fc2', rec)[0], 'needs_training_bridge')

    def test_dense(self):
        self.assertEqual(classify_record('model.layers.0.self_attn.qkv_proj', {'meta': {'matrix_shape': [6144, 4096]}})[0], 'dense_candidate')

    def test_expert(self):
        self.assertEqual(classify_record('model.layers.0.mlp.experts.7.down_proj', {'meta': {'matrix_shape': [4096, 4096]}})[0], 'needs_training_bridge')

    def test_ngram(self):
        self.assertEqual(classify_record('model.ngram_embedding.shard_0', {'meta': {'target_kind': 'cq_ngram_chunked_shard', 'matrix_shape': [4096,4096]}})[0], 'needs_training_bridge')

    def test_transpose(self):
        self.assertEqual(classify_record('model.layers.0.proj', {'meta': {'matrix_shape': [4096, 4096], 'transpose_source': True}})[0], 'needs_training_bridge')

if __name__ == '__main__':
    unittest.main()
