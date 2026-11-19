"""Optional calibrated boosted refinement; original RF and temporal paths stay active."""
import numpy as np
from .physical import PhysicalFeatures
class BoostedForest:
    def __init__(self,forest,booster):self.forest=forest;self.booster=booster
    @property
    def classes_(self):return self.forest.classes_
    @property
    def n_jobs(self):return self.forest.n_jobs
    @n_jobs.setter
    def n_jobs(self,value):self.forest.n_jobs=value
    def predict_proba(self,X):return self.forest.predict_proba(X)
    def boosted_probabilities(self,sequence,table):
        features=np.concatenate([PhysicalFeatures().transform(table),sequence.reshape(len(sequence),-1)],axis=1)
        return self.booster.predict_proba(features,thread_count=2)
