"""Same MSE minimax objective, centered conic form and independent certificates."""
import cvxpy as cp
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class CertifiedMaxRM:
    def __init__(self,seed,robust=True,n_estimators=16,max_depth=4,min_samples_leaf=12):
        self.seed,self.robust=seed,robust
        self.n_estimators,self.max_depth,self.min_samples_leaf=n_estimators,max_depth,min_samples_leaf

    def fit(self,X,y,env):
        rng=np.random.default_rng(self.seed);self.members_=[];self.certificates_=[];self.bootstrap_indices_=[]
        for _ in range(self.n_estimators):
            ids=rng.integers(len(y),size=len(y));self.bootstrap_indices_.append(ids)
            tree=DecisionTreeRegressor(max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,random_state=int(rng.integers(2**31-1))).fit(X[ids],y[ids])
            if self.robust:
                leaves,inv=np.unique(tree.apply(X[ids]),return_inverse=True);eb=env[ids];yb=y[ids];a=[];b=[];c=[]
                for e in np.unique(eb):
                    mask=eb==e;n=mask.sum()
                    a.append(np.bincount(inv[mask],minlength=len(leaves))/n)
                    b.append(np.bincount(inv[mask],weights=yb[mask],minlength=len(leaves))/n);c.append(np.mean(yb[mask]**2))
                a,b,c=np.array(a),np.array(b),np.array(c);theta=cp.Variable(len(leaves));t=cp.Variable(nonneg=True);constraints=[]
                for aa,bb,cc in zip(a,b,c):
                    center=np.divide(bb,np.sqrt(aa),out=np.zeros_like(bb),where=aa>0)
                    constraints.append(cp.sum_squares(cp.multiply(np.sqrt(aa),theta)-center)+float(cc-center@center)<=t)
                problem=cp.Problem(cp.Minimize(t),constraints);problem.solve(solver="CLARABEL")
                if problem.status!=cp.OPTIMAL or theta.value is None:
                    raise RuntimeError(f"Uncertified centered MaxRM solve: {problem.status}")
                values=np.asarray(theta.value)
                primal=max(float(np.mean((values[inv[eb==e]]-yb[eb==e])**2)) for e in np.unique(eb))
                weights=np.maximum(np.array([float(np.asarray(q.dual_value).reshape(-1)[0]) for q in constraints]),0.)
                if not np.isfinite(weights).all() or weights.sum()<=0:
                    raise RuntimeError('Missing finite dual certificate')
                weights/=weights.sum()
                aa=weights@a;bb=weights@b;lower=float(weights@c-np.sum(np.divide(bb**2,aa,out=np.zeros_like(bb),where=aa>0)))
                gap=primal-lower;residual=max(0.,primal-float(t.value));tol=1e-6*max(1.,abs(primal))
                before=max(float(np.mean((tree.predict(X[ids][eb==e])-yb[eb==e])**2)) for e in np.unique(eb))
                if not np.isfinite(values).all() or not np.isfinite([primal,lower,gap,residual,before]).all() or gap < -tol or gap>tol or residual>tol or primal>before+tol:
                    raise RuntimeError("MaxRM independent primal/dual/feasibility certificate failed")
                self.certificates_.append(dict(status=problem.status,primal=primal,dual_lower=lower,gap=gap,epigraph_violation=residual,baseline_worst=before))
                tree.tree_.value[leaves,0,0]=values
            self.members_.append(tree)
        return self

    def predict(self,X):
        return np.mean([tree.predict(X) for tree in self.members_],axis=0)
