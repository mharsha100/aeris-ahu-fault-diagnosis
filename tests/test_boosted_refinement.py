from pathlib import Path
import sys,tempfile,json,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.modeling import HybridPredictor,load_hybrid,sha256_file

class BoostedRefinementTests(unittest.TestCase):
    def model(self,parent,boost):
        class Forest:
            def predict_proba(self,x):return np.array([parent])
            def boosted_probabilities(self,s,t):return np.array([boost])
        cfg=dict(temperatures=dict(forest=1,temporal=1,hybrid=1),forest_weight=.5,review_confidence_threshold=.9,boosting=dict(weight=.9,temperature=1,hybrid_temperature=1,class_factors=[1.2,1,1,1]))
        return HybridPredictor(Forest(),None,cfg,{},dict(window_minutes=15,core_features=list(range(10)),tabular_features=list(range(66))),'cpu')
    def test_fusion_cannot_override_uncertain_parent_review(self):
        parent=[.2,.6,.1,.1];boost=[.001,.997,.001,.001]
        m=self.model(parent,boost)
        with patch('aeris.modeling.neural_logits',return_value=np.log([parent])):p=m.predict(np.zeros((1,15,10)),np.zeros((1,66)))
        expected=(.1*np.array(parent)+.9*np.array(boost))*[1.2,1,1,1];expected/=expected.sum()
        np.testing.assert_allclose(p['probabilities'][0],expected)
        self.assertGreater(p['confidence'][0],.9)
        self.assertFalse(p['branch_disagreement'][0])
        self.assertTrue(p['refinement_review'][0]);self.assertTrue(p['needs_review'][0])
    def test_confident_matching_parent_can_pass(self):
        parent=[.01,.97,.01,.01];m=self.model(parent,parent)
        with patch('aeris.modeling.neural_logits',return_value=np.log([parent])):p=m.predict(np.zeros((1,15,10)),np.zeros((1,66)))
        self.assertFalse(p['needs_review'][0])
    def test_invalid_boost_contract_fails_before_loading_artifacts(self):
        good=dict(weight=.9,temperature=1,hybrid_temperature=1,class_factors=[1,1,1,1])
        for change in [dict(weight=1),dict(weight=0),dict(temperature=0),dict(class_factors=[1,1,-1,1])]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);(p/'hybrid.json').write_text(json.dumps(dict(boosting={**good,**change})))
                (p/'READY.json').write_text(json.dumps(dict(hybrid_sha256=sha256_file(p/'hybrid.json'))))
                with self.assertRaisesRegex(ValueError,'boosted'):load_hybrid(p)
if __name__=='__main__':unittest.main()
