import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_layout import plan_records

class PaddingPlanTests(unittest.TestCase):
    def test_unaligned_vision_width(self):
        name = 'model.visual.blocks.0.mlp.linear_fc2'
        index = {'layers': {name: {'meta': {'matrix_shape': [1152, 4304]}}}}
        plans = plan_records(index, {'visual.blocks.0.mlp.linear_fc2': [1152, 4304]})
        self.assertEqual(plans[0]['shape'], [1152, 4304])
        self.assertEqual(((4304+31)//32)*32, 4320)
        self.assertEqual((1152*4320+1023)//1024, 4860)
        self.assertNotEqual(1152*((4320+1023)//1024), 4860)  # wrong per-row rounding

if __name__ == '__main__': unittest.main()
