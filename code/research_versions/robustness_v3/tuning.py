"""Source-only capacity pilot. Profiles are exploratory, not a best-method claim."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor
from nir_dg.series_v2.models import Model
from nir_dg.meta_forests import MetaForestsClassifier,normalized_weights
from .controlled_trees import DomainForest
from .maxrm import CertifiedMaxRM


def configurations(task):
    result=[]
    names=["rf","extra_trees","hist_gbdt","irf","anchor"]+(["crf","meta"] if task=="classification" else ["maxrm","era","directional_era"])
    for name in names:
        for profile in [0,1]:
            if name=="irf":
                variants=[dict(penalty=x) for x in [0.,.005,.05]]
            elif name=="crf":
                variants=[dict(lambda1=a,lambda2=b) for a,b in [(0.,0.),(.05,0.),(0.,.05),(.05,.05)]]
            elif name=="anchor":
                variants=[dict(gamma=x) for x in [1.,2.,8.]]
            elif name=="meta":
                variants=[dict(weighting=x) for x in ["uniform","accuracy","full"]]
            elif name=="maxrm":
                variants=[dict(robust=x) for x in [False,True]]
            elif name in {"era","directional_era"}:
                variants=[dict(strength=x) for x in [0.,.1,1.]]
            else:
                variants=[{}]
            for v in variants:
                result.append(dict(method=name,profile=profile,parameters=v))
    return result


class ConfiguredModel(Model):
    def fit(self,X,y,env):
        p=dict(self.params);profile=p.pop("profile");name,task,seed=self.name,self.task,self.seed
        self.mean_=float(np.mean(y)) if task=="regression" else 0.;self.scale_=max(float(np.std(y)),1e-8) if task=="regression" else 1.;z=(y-self.mean_)/self.scale_
        if name in {"rf","extra_trees"}:
            cls=({"rf":(RandomForestClassifier,RandomForestRegressor),"extra_trees":(ExtraTreesClassifier,ExtraTreesRegressor)}[name])[task=="regression"]
            self.model_=cls(n_estimators=[200,256][profile],max_depth=[None,12][profile],min_samples_leaf=[5,12][profile],max_features="sqrt" if task=="classification" else 1.,n_jobs=1,random_state=seed).fit(X,z)
        elif name=="hist_gbdt":
            cls=HistGradientBoostingClassifier if task=="classification" else HistGradientBoostingRegressor
            self.model_=cls(max_iter=[100,200][profile],max_leaf_nodes=[8,15][profile],min_samples_leaf=[12,24][profile],learning_rate=[.1,.05][profile],early_stopping=False,random_state=seed).fit(X,z)
        elif name in {"irf","crf"}:
            self.model_=DomainForest(task=task,mechanism=name,n_estimators=[16,64][profile],max_depth=[4,6][profile],min_samples_leaf=[12,24][profile],max_thresholds=[16,24][profile],random_state=seed,**p).fit(X,z,env)
        elif name=="maxrm":
            self.model_=CertifiedMaxRM(seed,robust=p["robust"],n_estimators=[16,32][profile],max_depth=[4,6][profile],min_samples_leaf=[12,24][profile]).fit(X,z,np.asarray(env))
        elif name=="meta":
            self.model_=MetaForestsClassifier(iterations=[3,5][profile],trees_per_forest=[4,8][profile],max_depth=[5,7][profile],sampling_ratio=[.7,1.][profile],feature_masking=False,random_state=seed).fit(X,z,env)
            if p["weighting"]=="uniform":
                self.model_.weights_=np.full(len(self.model_.members_),1/len(self.model_.members_))
            elif p["weighting"]=="accuracy":
                previous={};logs=[]
                for h in self.model_.history_:
                    d=h["train_domain"];w=previous.get(d,-np.log(len(np.unique(env))-1))+np.log(max(np.exp(h["accuracy"])-np.exp(.5),1e-8));previous[d]=w;logs.append(w)
                self.model_.weights_=normalized_weights(logs)
        elif name in {"era","directional_era"}:
            from erasplit.ensemble import EraHistGradientBoostingRegressor
            s=p["strength"];self.model_=EraHistGradientBoostingRegressor(gamma=s if name=="era" else 0.,blama=s if name=="directional_era" else 0.,vanna=1-s,max_iter=[80,160][profile],max_leaf_nodes=[8,15][profile],max_bins=32,min_samples_leaf=[12,24][profile],early_stopping=False,random_state=seed,n_jobs=1,learning_rate=[.1,.05][profile])
            _,e=np.unique(env,return_inverse=True);self.model_.fit(X,z,e.astype(np.int32))
        elif name=="anchor":
            from anchorboosting import AnchorBooster
            self.model_=AnchorBooster(gamma=p["gamma"],num_boost_round=[100,200][profile],objective="binary" if task=="classification" else "regression",alpha_cvar=1.,learning_rate=[.1,.05][profile],num_leaves=[8,15][profile],min_data_in_leaf=[12,24][profile],min_gain_to_split=1e-8,verbosity=-1,num_threads=1,seed=seed,deterministic=True,force_col_wise=True)
            _,e=np.unique(env,return_inverse=True);self.model_.fit(X,z,Z=np.eye(e.max()+1)[e])
        else:
            raise ValueError(name)
        return self
