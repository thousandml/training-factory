import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_layout import plan_records


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.prefix = 'model.layers.0.self_attn.'
        self.linears = {self.prefix + p: [m, 4096] for p, m in [('q_proj', 4096), ('k_proj', 1024), ('v_proj', 1024), ('o_proj', 4096)]}

    def index(self, name, shape, **meta):
        return {'layers': {name: {'filename': 'quantization_data/cq-00001.safetensors', 'meta': {'matrix_shape': shape, **meta}}}}

    def test_llama_gqa_fused(self):
        name = self.prefix + 'qkv_proj'
        p = plan_records(self.index(name, [6144, 4096]), self.linears)[0]
        self.assertEqual(p['destinations'], [self.prefix + s for s in ('q_proj', 'k_proj', 'v_proj')])
        self.assertEqual(p['slices'], [[0, 4096], [4096, 5120], [5120, 6144]])

    def test_metadata_authoritative_order(self):
        keys = [self.prefix + s + '.weight' for s in ('v_proj', 'q_proj', 'k_proj')]
        p = plan_records(self.index('custom.fused', [6144, 4096], source_state_keys=keys,
                        source_shapes=[[1024,4096],[4096,4096],[1024,4096]]), self.linears)[0]
        self.assertEqual(p['slices'], [[0,1024],[1024,5120],[5120,6144]])

    def test_split_legacy(self):
        index = {'layers': {self.prefix + p: {'meta': {'original_shape': [m,4096]}} for p,m in [('q_proj',4096),('k_proj',1024),('v_proj',1024)]}}
        self.assertEqual(len(plan_records(index, self.linears)), 3)

    def test_gate_up(self):
        prefix = 'model.layers.0.mlp.'
        p = plan_records(self.index(prefix+'gate_up_proj',[28672,4096]),
            {prefix+s:[14336,4096] for s in ('gate_proj','up_proj')})[0]
        self.assertEqual(p['slices'], [[0,14336],[14336,28672]])

    def test_wrapper_alias(self):
        p = plan_records(self.index('language_model.layers.0.attn.o_proj',[4096,4096]), self.linears)[0]
        self.assertEqual(p['destinations'], [self.prefix+'o_proj'])

    def test_wrong_shape(self):
        with self.assertRaises(ValueError):
            plan_records(self.index(self.prefix+'qkv_proj',[12288,4096]),self.linears)

    def test_overlap(self):
        index = self.index(self.prefix+'qkv_proj',[6144,4096])
        index['layers'][self.prefix+'q_proj']={'meta':{'matrix_shape':[4096,4096]}}
        with self.assertRaises(ValueError):
            plan_records(index,self.linears)

    def test_bad_source_shape(self):
        with self.assertRaises(ValueError):
            plan_records(self.index(self.prefix+'o_proj',[4096,4096],source_shapes=[[4096,8192]]),self.linears)

    def test_skip_unmodeled_vision_only_by_opt_in(self):
        index = self.index('model.visual.blocks.0.mlp.linear_fc2', [1152, 4304])
        index['layers'][self.prefix + 'o_proj'] = {
            'meta': {'matrix_shape': [4096, 4096]}}
        with self.assertRaises(ValueError):
            plan_records(index, self.linears)
        plans = plan_records(index, self.linears, skip_unmodeled_auxiliary=True)
        self.assertEqual([p['record'] for p in plans], [self.prefix + 'o_proj'])

    def test_present_vision_never_silently_skipped(self):
        index = self.index('model.visual.blocks.0.mlp.linear_fc2', [1152, 4304])
        linears = {**self.linears, 'visual.blocks.0.mlp.linear_fc2':[1152,4304]}
        plans = plan_records(index, linears, skip_unmodeled_auxiliary=True)
        self.assertEqual(plans[0]['shape'], [1152, 4304])

    def test_absent_mtp_skipped_even_in_multimodal_mode(self):
        index = self.index('model.mtp.fc', [4096, 4096])
        index['layers'][self.prefix + 'o_proj'] = {'meta': {'matrix_shape': [4096, 4096]}}
        plans = plan_records(index, self.linears)
        self.assertEqual([p['record'] for p in plans], [self.prefix + 'o_proj'])

    def test_present_mtp_is_strict(self):
        linears = {**self.linears, 'mtp.layers.0.self_attn.o_proj': [4096, 4096]}
        with self.assertRaises(ValueError):
            plan_records(self.index('model.mtp.fc', [4096, 4096]), linears)

    def test_namespace_protected(self):
        linears = {**self.linears, 'mtp.fc': [4096, 4096]}
        with self.assertRaises(ValueError):
            plan_records(self.index('model.mtp.layers.0.self_attn.o_proj',[4096,4096]), linears)

    def test_expert_rejected(self):
        with self.assertRaises(ValueError):
            plan_records(self.index('model.layers.0.mlp.experts.0.down_proj',[4096,4096],slice_index=0),self.linears)

    def test_ambiguous_alias(self):
        linears = dict(self.linears)
        linears['layers.0.attn.o_proj']=[4096,4096]
        with self.assertRaises(ValueError):
            plan_records(self.index('language_model.layers.0.attn.o_proj',[4096,4096]),linears)

    def test_nested_language_wrappers(self):
        from checkpoint_layout import resolve
        self.assertEqual(resolve('model.language_model.model.layers.0.self_attn.o_proj', self.linears),
                         self.prefix + 'o_proj')

    def test_wrapped_vision_not_backbone(self):
        from checkpoint_layout import namespace
        self.assertEqual(namespace('model.language_model.model.visual.blocks.0.mlp.fc2'), 'vision')
        self.assertEqual(namespace('model.language_model.model.mtp.layers.0.mlp.down_proj'), 'mtp')

    def test_audio_vision_never_match_language(self):
        from checkpoint_layout import resolve
        with self.assertRaises(ValueError):
            resolve('model.visual.layers.0.self_attn.o_proj', self.linears)

    def test_explicit_map(self):
        p=plan_records(self.index('renamed',[4096,4096]),self.linears,{'renamed': self.prefix+'o_proj'})[0]
        self.assertEqual(p['destinations'],[self.prefix+'o_proj'])

if __name__ == '__main__':
    unittest.main()
