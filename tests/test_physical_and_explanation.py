from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from aeris.physical import PhysicalFeatures
from aeris.operations import grouped_explanation
from aeris.modeling import create_temporal
class RepresentationAndExplanation(unittest.TestCase):
    def test_near_zero_temperature_span_is_finite(self):
        x=np.zeros((3,66),np.float32);x[:,[0,40]]=20
        y=PhysicalFeatures().transform(x);self.assertEqual(y.shape,(3,112));self.assertTrue(np.isfinite(y).all())
    def test_all_temporal_pooling_variants_keep_four_outputs(self):
        for kind in ['last','attention','combined']:
            m=create_temporal('physical_causal_tcn_'+kind,dict(input_features=10,channels=8,dropout=.1,dilations=[1,2,4],classes=4,raw_mean=[0]*10,raw_scale=[1]*10));self.assertEqual(m(torch.zeros(2,15,10)).shape,(2,4))
    def test_exact_grouped_shapley_additivity_and_repeatability(self):
        class Model:
            scaler={'mean':[0]*10}
            def predict_raw(self,x):
                score=np.asarray(x).mean((1,2));p=1/(1+np.exp(-score));return {'probabilities':np.column_stack([p,(1-p)/3,(1-p)/3,(1-p)/3])}
        raw=np.ones((15,10));raw[:,[4,9]]=0;m=Model();a=grouped_explanation(m,raw);b=grouped_explanation(m,raw)
        self.assertLess(a['additivity_error'],1e-12);self.assertEqual(a,b);self.assertEqual(a['coalitions_evaluated'],16)
if __name__=='__main__':unittest.main()
