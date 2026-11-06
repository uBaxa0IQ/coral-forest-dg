"""Analytic checks for aggregation and multiplicity; no experimental target reads."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_series_v2 import sign_test,holm,analysis,expand_target_row


def test_exact_sign_test_and_holm_known_values():
    assert sign_test(np.ones(10))==(2/1024,10)
    assert sign_test(np.r_[np.ones(5),-np.ones(5)])[0]==1.
    assert sign_test(np.zeros(10))==(1.,0)
    result=holm(pd.DataFrame({"p_value":[.04,.001,.01]}))
    assert np.allclose(result.p_holm,[.04,.003,.02])
    row=expand_target_row({"n":8192,"k":8},{"n":2000,"method":"irf"},1.)
    assert row["n"]==8192 and row["target_rows"]==2000
    incomplete=holm(pd.DataFrame({"p_value":[.001,np.nan,.01]}))
    assert np.allclose(incomplete.p_holm.iloc[[0,2]],[.003,.02])
    assert np.isnan(incomplete.p_holm.iloc[1])


def test_targets_seeds_and_replicas_have_correct_weights():
    methods={"classification":["irf","crf","meta","anchor"],"regression":["irf","maxrm","era","directional_era","anchor"]}
    target=[];cv=[]
    for task,names in methods.items():
        for k,n,imb in [(4,8192,1),(8,8192,1),(16,8192,1),(8,2048,1),(8,8192,8)]:
            for rep in range(10):
                for seed in [11,47]:
                    job=f"{task}_{k}_{n}_{imb}_{rep}_{seed}"
                    common=dict(kind="synthetic",dataset="synthetic",task=task,scenario="spurious",k=k,n=n,imbalance=imb,replica=rep,job_id=job,model_seed=seed)
                    for method in names:
                        for ci in [0,1]:
                            for domain,count,gain in [("small",10,.02),("large",10000,.06)]:
                                g=gain*(k/8)*np.sqrt(n/8192)*(2 if imb==8 else 1)*ci
                                selected=ci==(1 if seed==11 else 0)
                                auc=.6+g if task=="classification" else np.nan
                                mse=100*(1-g) if task=="regression" else np.nan
                                primary=auc if task=="classification" else -mse/100
                                target.append({**common,"method":method,"config":ci,"target_domain":domain,"n_observations":count,"selected":selected,"source_y_scale":10.,"primary":primary,"roc_auc":auc,"mse":mse,"rmse":np.sqrt(mse),"mae":np.nan,"log_loss":np.nan,"balanced_accuracy":np.nan,"average_precision":np.nan})
                            cv.append(dict(job_id=job,method=method,config=ci,fold=0,primary=.7 if ci==(1 if seed==11 else 0) else .6))
    a=analysis(pd.DataFrame(target),pd.DataFrame(cv),{"statistics":{"primary_methods":methods}})
    h=a["hypotheses"]
    assert len(h)==18
    # Equal domain means .04, then equal seed mean .02 at K8. Counting rows
    # would approach .06 and treating seeds as independent would give n=20.
    assert np.allclose(h[h.hypothesis=="H1"]["mean"],.03)
    assert np.allclose(h[h.hypothesis=="H2"]["mean"],.02)
    assert (h.replicas==10).all()
    assert np.allclose(h.p_holm,9*2/1024)
    assert (a["selection_diagnostics"].retrospective_oracle_regret>=-1e-15).all()
    # Losing one seed must remove that replica-method cell, not quietly reweight
    # the remaining seed or reduce the family size for the other eight tests.
    frame=pd.DataFrame(target)
    missing=(frame.task=="regression")&(frame.method=="maxrm")&(frame.k==8)&(frame.imbalance==8)&(frame.replica==3)&(frame.model_seed==47)
    b=analysis(frame[~missing],pd.DataFrame(cv),{"statistics":{"primary_methods":methods}})
    row=b["hypotheses"][(b["hypotheses"].hypothesis=="H2")&(b["hypotheses"].method=="maxrm")].iloc[0]
    assert row.replicas==9 and row.status=="incomplete_source_solver_failure"
    assert np.isnan(row.p_value) and np.isnan(row.p_holm)
