from pathlib import Path
import sys,unittest,tempfile,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from aeris.operations import Ledger
spec=importlib.util.spec_from_file_location('feedback_candidate',ROOT/'scripts/14_feedback_candidate.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class FeedbackEligibility(unittest.TestCase):
    def test_empty_and_simulated_labels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'db.sqlite3')
            with self.assertRaisesRegex(ValueError,'No independently'):module.validate_records(ledger,{'reviewed_by':'test reviewer','records':[]},pd.DataFrame(),set())
            p=ledger.save('id','hash','test','Unknown','v0',{},dict(status='review_required',class_id=0),[[1]*10]*15,.1)
            ledger.feedback(p['prediction_id'],dict(verified_class='normal',inspector='test',provenance='simulated inspection fixture'))
            manifest={'reviewed_by':'test reviewer','records':[dict(prediction_id=p['prediction_id'],window_id='w',independent_case_id='c',partition='train')]}
            with self.assertRaisesRegex(ValueError,'Test, simulated'):module.validate_records(ledger,manifest,pd.DataFrame({'condition_group':['day']},index=['w']),{'w'})
if __name__=='__main__':unittest.main()
