"""Deterministic, deployment-available temperature/command relationships.

No labels, timestamps, true actuator positions, or dataset identifiers are used.
Ratios are diagnostic features, not estimates of measured damper position.
"""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class PhysicalFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):return self
    def transform(self,X):
        X=np.asarray(X,dtype=np.float32);parts=[X]
        for start in [0,40]:
            s,o,m,r,fan,speed,d,c,h,occ=X[:,start:start+10].T
            span=o-r;safe=np.where(np.abs(span)>=2,span,np.where(span<0,-2,2))
            mix_ratio=np.clip((m-r)/safe,-2,3)
            coil=s-m;mix=m-(d*o+(1-d)*r)
            parts.extend([np.column_stack([coil,s-r,m-r,span,mix,mix_ratio,mix_ratio-d,
                np.abs(span)<2,np.clip(coil/(c+.1),-100,100),np.clip(coil/(h+.1),-100,100),
                coil*c,coil*h,coil*(1-c),coil*(1-h),mix*d,mix*(1-d),c*h,
                c<.05,h<.05,d<.05,d>.95,coil*speed,(m-r)*speed])])
        return np.nan_to_num(np.concatenate(parts,axis=1),nan=0,posinf=1e6,neginf=-1e6).astype(np.float32)

class PhysicalForest(BaseEstimator):
    """Serializable RF wrapper retaining the original 66-feature external contract."""
    def __init__(self,forest):self.forest=forest;self.features=PhysicalFeatures()
    @property
    def n_jobs(self):return self.forest.n_jobs
    @n_jobs.setter
    def n_jobs(self,value):self.forest.n_jobs=value
    @property
    def classes_(self):return self.forest.classes_
    def fit(self,X,y):self.forest.fit(self.features.transform(X),y);return self
    def predict_proba(self,X):return self.forest.predict_proba(self.features.transform(X))

