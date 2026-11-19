from pathlib import Path
import sys,tempfile,json,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.modeling import load_hybrid,sha256_file
class BundleContractTests(unittest.TestCase):
    def test_reject_single_branch_even_with_valid_manifest_hash(self):
        original=json.loads((ROOT/'data/aeris_models/v1_attention/hybrid.json').read_text())
        for weight in [0,1]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);cfg=dict(original,forest_weight=weight);(p/'hybrid.json').write_text(json.dumps(cfg));(p/'READY.json').write_text(json.dumps({'hybrid_sha256':sha256_file(p/'hybrid.json')}))
                with self.assertRaisesRegex(ValueError,'both RF and temporal'):load_hybrid(p)
    def test_config_hash_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'hybrid.json').write_text('{}');(p/'READY.json').write_text(json.dumps({'hybrid_sha256':'wrong'}))
            with self.assertRaisesRegex(ValueError,'configuration changed'):load_hybrid(p)
if __name__=='__main__':unittest.main()
