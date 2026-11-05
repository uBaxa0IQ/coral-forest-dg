"""Bounded source-only engineering diagnostics. Never evaluates target quality."""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[k]="1"
from hashlib import sha256
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from nir_dg.preprocessing import make_preprocessor
from nir_dg.series_v2.data import source_real,features
from nir_dg.series_v2.generators import source_frame
from nir_dg.series_v2.engine import ROOT,RUN,verify_freeze,pack_folds,write_json,now
from research_versions.robustness_v3.controlled_trees import DomainForest
from research_versions.robustness_v3.preprocessing import make_preprocessor as canonical_preprocessor

OUT=ROOT/"reports/closeout/source_diagnostics"


def tree_invariants(model,X,env):
    counts=dict(empty_nodes=0,nonfinite_nodes=0,small_leaves=0,support_violations=0)
    def visit(node,ids):
        counts["empty_nodes"]+=int(len(ids)==0);counts["nonfinite_nodes"]+=int(not np.isfinite(node.value))
        if node.feature<0:
            counts["small_leaves"]+=int(len(ids)<model.min_samples_leaf);return
        left=X[ids,node.feature]<=node.threshold
        counts["support_violations"]+=int(set(env[ids[left]])!=set(env[ids]) or set(env[ids[~left]])!=set(env[ids]))
        visit(node.left,ids[left]);visit(node.right,ids[~left])
    for tree,ids in zip(model.trees_,model.bootstrap_indices_):
        visit(tree,ids)
    return counts


def irf_and_null_checks(protocol):
    checks=pd.read_csv(ROOT/"reports/series_v2/audit/tree_numeric_checks.csv")
    bad=checks[checks.iloc[:,2:].sum(axis=1)>0]
    ids=set(bad.job_id);specs=[s for s in protocol["jobs"] if s["job_id"] in ids]
    results=[]
    for s in specs:
        f=source_real(s);p=make_preprocessor(f[features(f)]);X=p.fit_transform(f[features(f)]);y=f.__y.to_numpy();env=f.__env.to_numpy();y=(y-y.mean())/y.std()
        bootstrap=[]
        for penalty in [0.,.05,.5]:
            m=DomainForest(task="regression",penalty=penalty,random_state=s["model_seed"]).fit(X,y,env)
            values=tree_invariants(m,X,env);assert not any(values.values())
            assert np.isfinite(m.predict_values(X)).all()
            bootstrap.append([sha256(a.tobytes()).hexdigest() for a in m.bootstrap_indices_])
            results.append(dict(job_id=s["job_id"],penalty=penalty,source_rows=len(f),**values,train_predictions_finite=True))
        assert bootstrap[0]==bootstrap[1]==bootstrap[2]
    s=next(s for s in protocol["jobs"] if s["dataset"]=="beijing_pm25")
    f=source_real(s)[features(source_real(s))];a=f.copy();b=f.copy()
    a.loc[a.wd.isna(),"wd"]=np.nan;b.loc[b.wd.isna(),"wd"]=None
    p=canonical_preprocessor(a);xa=p.fit_transform(a);xb=p.transform(b)
    assert np.array_equal(xa,xb)
    pd.DataFrame(results).to_csv(OUT/"corrected_irf_source_checks.csv",index=False)
    return dict(corrected_irf_fits=len(results),affected_source_jobs=len(specs),all_tree_invariants_passed=True,bootstrap_same_across_penalties=True,beijing_source_rows=len(f),beijing_source_missing_rows=int(f.wd.isna().sum()),canonical_cache_transform_max_abs_difference=float(np.max(np.abs(xa-xb))),target_evaluations=0)


def certificates(a,b,c,theta,t,constraints,inverse,y,env):
    if theta.value is None:
        return dict(primal=None,dual_lower_bound=None,gap=None)
    values=np.asarray(theta.value)
    direct=np.array([np.mean((values[inverse[env==e]]-y[env==e])**2) for e in np.unique(env)])
    algebra=a@(values**2)-2*b@values+c
    assert np.allclose(direct,algebra,rtol=1e-10,atol=1e-10)
    weights=np.array([float(np.asarray(q.dual_value).reshape(-1)[0]) if q.dual_value is not None else 0. for q in constraints])
    weights=np.maximum(weights,0.)
    if weights.sum()<=0:
        return dict(primal=float(direct.max()),dual_lower_bound=None,gap=None)
    weights/=weights.sum();aa=weights@a;bb=weights@b;cc=weights@c
    active=aa>0
    lower=float(cc-np.sum(bb[active]**2/aa[active]));primal=float(direct.max())
    return dict(primal=primal,dual_lower_bound=lower,gap=primal-lower,epigraph_violation=float(max(0.,primal-float(t.value))),direct_vs_stats_max_abs=float(np.max(np.abs(direct-algebra))),active_dual_domains=int((weights>1e-5).sum()),minimum_weighted_leaf_curvature=float(aa.min()))


def maxrm_failure_checks(protocol):
    failures=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((RUN/"failures").glob("*.json"))]
    results=[];original_solves=0
    for failure in failures:
        s=failure["spec"];f=source_frame(s);y=f.__y.to_numpy();env=f.__env.to_numpy();stages=[]
        for fi,(tr,va,X,V) in enumerate(pack_folds(f,s)):
            stages.append((f"fold_{fi}",X,y[tr],env[tr]))
        stages.append(("full",make_preprocessor(f[features(f)]).fit_transform(f[features(f)]),y,env))
        found=False
        for stage,X,yy,ee in stages:
            z=(yy-yy.mean())/yy.std();rng=np.random.default_rng(s["model_seed"])
            for ti in range(16):
                ids=rng.integers(len(z),size=len(z));tree=DecisionTreeRegressor(max_depth=4,min_samples_leaf=12,random_state=int(rng.integers(2**31-1))).fit(X[ids],z[ids])
                leaves,inverse=np.unique(tree.apply(X[ids]),return_inverse=True);a=[];b=[];c=[];eb=ee[ids];zb=z[ids]
                for e in np.unique(eb):
                    mask=eb==e;n=mask.sum();a.append(np.bincount(inverse[mask],minlength=len(leaves))/n);b.append(np.bincount(inverse[mask],weights=zb[mask],minlength=len(leaves))/n);c.append(np.mean(zb[mask]**2))
                a,b,c=np.array(a),np.array(b),np.array(c)
                theta=cp.Variable(len(leaves));t=cp.Variable(nonneg=True)
                cons=[cp.sum(cp.multiply(ae,cp.square(theta)))-2*be@theta+ce<=t for ae,be,ce in zip(a,b,c)]
                problem=cp.Problem(cp.Minimize(t),cons);problem.solve(solver="CLARABEL");original_solves+=1
                if problem.status==cp.OPTIMAL:
                    continue
                base=dict(job_id=s["job_id"],scenario=s["scenario"],k=s["k"],n=s["n"],replica=s["replica"],model_seed=s["model_seed"],stage=stage,tree_index=ti,leaf_count=len(leaves))
                results.append({**base,"diagnostic":"original_default","status":problem.status,**certificates(a,b,c,theta,t,cons,inverse,zb,eb)})
                for name,solver,opts,centered in [("clarabel_strict","CLARABEL",dict(max_iter=1000,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9),False),("clarabel_centered","CLARABEL",{},True),("scs_independent","SCS",dict(eps=1e-7,max_iters=200000),True)]:
                    th=cp.Variable(len(leaves));tt=cp.Variable(nonneg=True);cc=[]
                    for ae,be,ce in zip(a,b,c):
                        if centered:
                            active=ae>0;mean=np.divide(be,np.sqrt(ae),out=np.zeros_like(be),where=active)
                            expression=cp.sum_squares(cp.multiply(np.sqrt(ae),th)-mean)+float(ce-np.sum(mean**2))
                        else:
                            expression=cp.sum(cp.multiply(ae,cp.square(th)))-2*be@th+ce
                        cc.append(expression<=tt)
                    pr=cp.Problem(cp.Minimize(tt),cc)
                    try:
                        pr.solve(solver=solver,**opts)
                        results.append({**base,"diagnostic":name,"status":pr.status,**certificates(a,b,c,th,tt,cc,inverse,zb,eb)})
                    except Exception as error:
                        results.append({**base,"diagnostic":name,"status":"error","error":str(error)})
                found=True;break
            if found:
                break
        if not found:
            results.append(dict(job_id=s["job_id"],diagnostic="original_default",status="not_reproduced_in_this_diagnostic"))
        print("MAXRM SOURCE CHECK",s["job_id"],found,flush=True)
    pd.DataFrame(results).to_csv(OUT/"maxrm_numerical_certificates.csv",index=False)
    return dict(original_tree_solves=original_solves,source_failure_cases=len(failures),historical_results_replaced=False,target_evaluations=0)


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True);frozen=verify_freeze()
    result=dict(at=now(),protocol_sha256=frozen["sha256"],irf_and_null=irf_and_null_checks(frozen["protocol"]),maxrm=maxrm_failure_checks(frozen["protocol"]),scope="source-only engineering validation; no generalization claim")
    write_json(OUT/"summary.json",result);print(json.dumps(result,indent=2),flush=True)
