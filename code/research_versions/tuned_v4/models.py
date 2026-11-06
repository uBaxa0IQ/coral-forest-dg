"""Explicit hyperparameters and capacity-matched mechanism controls."""
import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor
from nir_dg.series_v2.models import Model
from nir_dg.meta_forests import MetaForestsClassifier
from research_versions.robustness_v3.controlled_trees import DomainForest
from research_versions.robustness_v3.maxrm import CertifiedMaxRM


class TunedModel(Model):
    def fit(self,X,y,env):
        p=dict(self.params);task=self.task;name=self.name
        self.mean_=float(np.mean(y)) if task=='regression' else 0.
        self.scale_=max(float(np.std(y)),1e-8) if task=='regression' else 1.
        z=(np.asarray(y)-self.mean_)/self.scale_
        if name in {'rf','extra_trees'}:
            cls={'rf':(RandomForestClassifier,RandomForestRegressor),'extra_trees':(ExtraTreesClassifier,ExtraTreesRegressor)}[name][task=='regression']
            self.model_=cls(**p,n_jobs=1,random_state=self.seed).fit(X,z)
        elif name=='hist_gbdt':
            cls=HistGradientBoostingClassifier if task=='classification' else HistGradientBoostingRegressor
            self.model_=cls(**p,early_stopping=False,random_state=self.seed).fit(X,z)
        elif name in {'irf','crf'}:
            self.model_=DomainForest(task=task,mechanism=name,random_state=self.seed,**p).fit(X,z,env)
        elif name=='maxrm':
            self.model_=CertifiedMaxRM(self.seed,**p).fit(X,z,np.asarray(env))
        elif name=='meta':
            self.model_=MetaForestsClassifier(**p,random_state=self.seed).fit(X,z,env)
        elif name in {'era','directional_era'}:
            from erasplit.ensemble import EraHistGradientBoostingRegressor
            s=p.pop('strength');_,e=np.unique(env,return_inverse=True)
            self.model_=EraHistGradientBoostingRegressor(gamma=s if name=='era' else 0.,blama=s if name=='directional_era' else 0.,vanna=1-s,**p,early_stopping=False,n_jobs=1,random_state=self.seed).fit(X,z,e.astype(np.int32))
        elif name=='anchor':
            from anchorboosting import AnchorBooster
            _,e=np.unique(env,return_inverse=True)
            self.model_=AnchorBooster(**p,objective='binary' if task=='classification' else 'regression',alpha_cvar=1.,min_gain_to_split=1e-8,verbosity=-1,num_threads=1,seed=self.seed,deterministic=True,force_col_wise=True).fit(X,z,Z=np.eye(e.max()+1)[e])
        else:raise ValueError(name)
        return self

    def uniform_control(self):
        if self.name!='meta':raise ValueError('Only Meta can reuse its identical members')
        result=copy.copy(self);result.model_=copy.copy(self.model_)
        result.model_.weights_=np.full(len(self.model_.members_),1/len(self.model_.members_))
        return result


def control_parameters(method,p):
    p=dict(p)
    if method=='irf':p['penalty']=0.
    elif method=='crf':p.update(lambda1=0.,lambda2=0.)
    elif method=='maxrm':p['robust']=False
    elif method in {'era','directional_era'}:p['strength']=0.
    elif method=='anchor':p['gamma']=1.
    elif method=='meta':raise ValueError('Use literally the same trained members')
    else:return None
    return p
