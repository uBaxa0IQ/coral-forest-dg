"""Registered v2 controls. Structural parameters never vary inside a pair."""
import numpy as np
import cvxpy as cp
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from nir_dg.meta_forests import MetaForestsClassifier, normalized_weights
from .controlled_trees import DomainForest


def grids(task):
    g={"rf": [{}], "extra_trees": [{}], "hist_gbdt": [{}],
       "irf": [{"penalty": x} for x in [0., .05, .5]],
       "anchor": [{"gamma": x} for x in [1., 2., 8.]]}
    if task=="classification":
        g["crf"]=[{"lambda1":x,"lambda2":x} for x in [0., .05, .5]]
        g["meta"]=[{"weighting": x, "masking": False} for x in ["uniform","accuracy","full"]]
        g["meta_masked"]=[{"weighting": x, "masking": True} for x in ["uniform","accuracy","full"]]
    else:
        g["maxrm"]=[{"robust":False},{"robust":True}]
        g["era"]=[{"strength":x} for x in [0., .1, 1.]]
        g["directional_era"]=[{"strength":x} for x in [0., .1, 1.]]
    return g


class EfficientMaxRM:
    """Exact bootstrap empirical MSE epigraph via leaf sufficient statistics.

    sum_i(theta_leaf(i)-y_i)^2 / n_e equals
    sum_l a_el*theta_l^2 - 2*b_el*theta_l + c_e.
    The number of cone coordinates is now leaves, rather than training rows.
    """
    def __init__(self, seed, robust=True, trees=16, depth=4):
        self.seed,self.robust,self.trees,self.depth=seed,robust,trees,depth

    def fit(self,X,y,env):
        rng=np.random.default_rng(self.seed)
        self.members_,self.optimization_,self.bootstrap_indices_=[],[],[]
        for _ in range(self.trees):
            ids=rng.integers(len(y),size=len(y))
            tree=DecisionTreeRegressor(max_depth=self.depth,min_samples_leaf=12,random_state=int(rng.integers(2**31-1)))
            tree.fit(X[ids],y[ids]); self.bootstrap_indices_.append(ids)
            if self.robust:
                leaves,inverse=np.unique(tree.apply(X[ids]),return_inverse=True)
                a,b,c=[],[],[]
                for e in np.unique(env[ids]):
                    mask=env[ids]==e; ne=mask.sum()
                    a.append(np.bincount(inverse[mask],minlength=len(leaves))/ne)
                    b.append(np.bincount(inverse[mask],weights=y[ids][mask],minlength=len(leaves))/ne)
                    c.append(float(np.mean(y[ids][mask]**2)))
                a,b,c=np.array(a),np.array(b),np.array(c)
                theta=cp.Variable(len(leaves)); t=cp.Variable(nonneg=True)
                problem=cp.Problem(cp.Minimize(t),[cp.sum(cp.multiply(ae,cp.square(theta)))-2*be@theta+ce<=t for ae,be,ce in zip(a,b,c)])
                problem.solve(solver="CLARABEL")
                if problem.status != cp.OPTIMAL or theta.value is None:
                    raise RuntimeError(f"Unverified MaxRM optimum: {problem.status}")
                before_values=tree.tree_.value[leaves,0,0]
                before=float(np.max(a@before_values**2-2*b@before_values+c))
                after=float(np.max(a@theta.value**2-2*b@theta.value+c))
                if after>before+1e-5*max(1.,before):
                    raise RuntimeError("MaxRM worsened a feasible baseline")
                tree.tree_.value[leaves,0,0]=theta.value
                self.optimization_.append({"before":before,"after":after,"status":problem.status})
            self.members_.append(tree)
        return self

    def predict(self,X):
        return np.mean([t.predict(X) for t in self.members_],axis=0)


class Model:
    def __init__(self,name,task,seed,params=None):
        self.name,self.task,self.seed,self.params=name,task,seed,dict(params or {})

    def fit(self,X,y,env):
        name,task,p,seed=self.name,self.task,self.params,self.seed
        self.mean_=float(np.mean(y)) if task=="regression" else 0.
        self.scale_=max(float(np.std(y)),1e-8) if task=="regression" else 1.
        z=(np.asarray(y)-self.mean_)/self.scale_
        if name in {"rf","extra_trees"}:
            cls=({"rf":(RandomForestClassifier,RandomForestRegressor),"extra_trees":(ExtraTreesClassifier,ExtraTreesRegressor)}[name])[task=="regression"]
            self.model_=cls(n_estimators=200,min_samples_leaf=5,max_features="sqrt" if task=="classification" else 1.,n_jobs=1,random_state=seed).fit(X,z)
        elif name=="hist_gbdt":
            cls=HistGradientBoostingClassifier if task=="classification" else HistGradientBoostingRegressor
            self.model_=cls(max_iter=200,max_leaf_nodes=15,min_samples_leaf=12,early_stopping=False,random_state=seed).fit(X,z)
        elif name in {"irf","crf"}:
            self.model_=DomainForest(task=task,mechanism=name,random_state=seed,**p).fit(X,z,env)
        elif name.startswith("meta"):
            self.model_=MetaForestsClassifier(iterations=3,trees_per_forest=4,max_depth=5,feature_masking=p["masking"],random_state=seed).fit(X,z,env)
            # Reweight EXACTLY the same trained members; training is independent
            # of previous_weights. Masking is a distinct registered family.
            self.model_.full_weights_=self.model_.weights_.copy()
            if p["weighting"]=="uniform":
                self.model_.weights_=np.full(len(self.model_.members_),1/len(self.model_.members_))
            elif p["weighting"]=="accuracy":
                previous={}; logs=[]
                for h in self.model_.history_:
                    domain=h["train_domain"]
                    log=previous.get(domain,-np.log(len(np.unique(env))-1))+np.log(max(np.exp(h["accuracy"])-np.exp(.5),1e-8))
                    previous[domain]=log;logs.append(log)
                self.model_.weights_=normalized_weights(logs)
        elif name=="maxrm":
            self.model_=EfficientMaxRM(seed,p["robust"]).fit(X,z,np.asarray(env))
        elif name in {"era","directional_era"}:
            from erasplit.ensemble import EraHistGradientBoostingRegressor
            s=p["strength"]
            self.model_=EraHistGradientBoostingRegressor(gamma=s if name=="era" else 0.,blama=s if name=="directional_era" else 0.,vanna=1.-s,max_iter=80,max_leaf_nodes=8,max_bins=32,min_samples_leaf=12,early_stopping=False,random_state=seed,n_jobs=1,learning_rate=.1)
            _,e=np.unique(env,return_inverse=True)
            self.model_.fit(X,z,e.astype(np.int32))
        elif name=="anchor":
            from anchorboosting import AnchorBooster
            self.model_=AnchorBooster(gamma=p["gamma"],num_boost_round=100,objective="binary" if task=="classification" else "regression",alpha_cvar=1.,learning_rate=.1,num_leaves=8,min_data_in_leaf=12,min_gain_to_split=1e-8,verbosity=-1,num_threads=1,seed=seed,deterministic=True,force_col_wise=True)
            _,e=np.unique(env,return_inverse=True)
            self.model_.fit(X,z,Z=np.eye(e.max()+1)[e])
        else:
            raise ValueError(name)
        return self

    def predict(self,X):
        if self.task=="classification" and self.name!="anchor":
            p=self.model_.predict_proba(X)[:,1]
        else:
            p=self.model_.predict(X)
        return np.asarray(p)*self.scale_+self.mean_
