"""Registered analysis; reads immutable results, never fits or selects on target."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[key]="1"
from hashlib import sha256
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t,spearmanr
from nir_dg.series_v2.engine import ROOT,RUN,verify_freeze,write_json,now
from nir_dg.series_v2.data import source_real
from nir_dg.series_v2.generators import source_frame

OUT=ROOT/"reports/series_v2"
TABLES=OUT/"tables"
DIMS=["kind","dataset","task","scenario","k","n","imbalance","replica"]
ROBUST=["irf","crf","meta","meta_masked","anchor","maxrm","era","directional_era"]


def interval(values):
    v=np.asarray(values,float);n=len(v);mean=float(v.mean())
    half=float(student_t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)) if n>1 else 0.
    return dict(replicas=n,mean=mean,ci_low=mean-half,ci_high=mean+half)


def sign_test(values):
    v=np.asarray(values);v=v[v!=0];n=len(v)
    if n==0:
        return 1.,0
    tail=min(int((v>0).sum()),int((v<0).sum()))
    return min(1.,2*sum(math.comb(n,i) for i in range(tail+1))/2**n),n


def holm(frame):
    frame=frame.copy();missing=frame.p_value.isna();p=frame.p_value.fillna(1.).to_numpy();order=np.argsort(p,kind="stable")
    corrected=np.minimum(1.,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    frame["p_holm"]=0.;frame.loc[frame.index[order],"p_holm"]=corrected
    frame.loc[missing,"p_holm"]=np.nan
    return frame


def expand_target_row(spec,row,scale):
    values=dict(row)
    values["target_rows"]=values.pop("n")
    # Source training N and target sample count are different quantities.
    # Keep the registered factorial key intact when flattening nested JSON.
    return {**spec,"source_y_scale":scale,**values}


def collect():
    frozen=verify_freeze();target=[];cv=[];timings=[];inventory=[];source_scales={}
    for spec in frozen["protocol"]["jobs"]:
        path=RUN/"jobs"/spec["job_id"]
        r=json.loads((path/"result.json").read_text(encoding="utf-8"));s=json.loads((path/"source_selection.json").read_text(encoding="utf-8"))
        if spec["context_id"] not in source_scales:
            frame=source_frame(spec) if spec["kind"]=="synthetic" else source_real(spec)
            source_scales[spec["context_id"]]=max(float(np.std(frame.__y.to_numpy(),ddof=0)),1e-8)
        actual_scale=source_scales[spec["context_id"]]
        assert np.isclose(actual_scale,s["source_y_scale"],rtol=1e-12,atol=1e-12)
        assert r["source_y_scale"]==s["source_y_scale"]
        for row in r["metrics"]:
            target.append(expand_target_row(spec,row,r["source_y_scale"]))
        for row in s["cv"]:
            cv.append({**spec,**row})
        for row in s["times"]:
            timings.append({**spec,**row})
        inventory.append(dict(job_id=spec["job_id"],status=r["status"],omitted_methods=";".join(r.get("omitted_methods",[])),source_rows=s["source_rows"],source_fits=len(s["times"]),repeated_fits=len(s["replays"]),source_seconds=s["source_seconds"],models=len(s["times"])//3,source_scale_recalculated=actual_scale,source_scale_recorded=r["source_y_scale"]))
    target=pd.DataFrame(target);cv=pd.DataFrame(cv)
    target["primary"]=np.where(target.task=="classification",target.roc_auc,-target.mse/target.source_y_scale**2)
    return frozen,target,cv,pd.DataFrame(timings),pd.DataFrame(inventory)


def analysis(target,cv,protocol):
    selected=target[target.selected].copy()
    curves=target[target.method.isin(ROBUST)].copy()
    control_rep=curves[curves.config==0].groupby(DIMS+["method"],as_index=False).agg(primary=("primary","mean"),roc_auc=("roc_auc","mean"),rmse=("rmse","mean"),mse=("mse","mean"),seed_count=("model_seed","nunique"))
    control_rep=control_rep[control_rep.seed_count==2]
    control_metrics=control_rep.groupby(DIMS[:-1]+["method"],as_index=False)[["primary","roc_auc","rmse","mse"]].mean()
    control_metrics=control_metrics.rename(columns={c:"control_"+c for c in ["primary","roc_auc","rmse","mse"]})
    base=curves[curves.config==0][["job_id","method","target_domain","primary"]].rename(columns={"primary":"control_primary"})
    curves=curves.merge(base,on=["job_id","method","target_domain"],validate="many_to_one")
    curves["gain"]=curves.primary-curves.control_primary
    paired=curves[curves.selected].copy()
    units=paired.groupby(DIMS+["method"],as_index=False).agg(gain=("gain","mean"),seed_count=("model_seed","nunique"))
    units=units[units.seed_count==2].copy()
    macro=selected.groupby(DIMS+["job_id","model_seed","method"],as_index=False).agg(primary=("primary","mean"),roc_auc=("roc_auc","mean"),rmse=("rmse","mean"),mse=("mse","mean"),mae=("mae","mean"),log_loss=("log_loss","mean"),balanced_accuracy=("balanced_accuracy","mean"),average_precision=("average_precision","mean"),source_y_scale=("source_y_scale","first"))
    worst=target.groupby(["job_id","method","config"],as_index=False).primary.min().rename(columns={"primary":"worst_primary"})
    zero=worst[worst.config==0][["job_id","method","worst_primary"]].rename(columns={"worst_primary":"control_worst"})
    chosen=selected[["job_id","method","config"]].drop_duplicates().merge(worst,on=["job_id","method","config"]).merge(zero,on=["job_id","method"])
    chosen["worst_gain"]=chosen.worst_primary-chosen.control_worst
    macro=macro.merge(chosen[["job_id","method","worst_primary","worst_gain"]],on=["job_id","method"],validate="one_to_one")
    confirmations=[];contrast_rows=[]
    syn=units[units.kind=="synthetic"]
    for hypothesis in ["H1","H2"]:
        candidates=[]
        for task,methods in protocol["statistics"]["primary_methods"].items():
            for method in methods:
                f=syn[(syn.task==task)&(syn.method==method)&(syn.scenario=="spurious")&(syn.n==8192)]
                if hypothesis=="H1":
                    a=f[(f.k==4)&(f.imbalance==1)];b=f[(f.k==16)&(f.imbalance==1)]
                else:
                    a=f[(f.k==8)&(f.imbalance==1)];b=f[(f.k==8)&(f.imbalance==8)]
                joined=a[["replica","gain"]].merge(b[["replica","gain"]],on="replica",suffixes=("_a","_b"),validate="one_to_one")
                assert len(joined)<=10
                d=joined.gain_b-joined.gain_a;p,ne=sign_test(d)
                complete=len(joined)==10
                candidates.append(dict(hypothesis=hypothesis,task=task,method=method,**interval(d),p_value=p if complete else np.nan,descriptive_complete_case_p=p,nonzero_replicas=ne,status="complete" if complete else "incomplete_source_solver_failure",planned_replicas=10))
                for rep,x in zip(joined.replica,d):
                    contrast_rows.append(dict(hypothesis=hypothesis,task=task,method=method,replica=int(rep),gain_difference=float(x)))
        confirmations.append(holm(pd.DataFrame(candidates)))
    hypotheses=pd.concat(confirmations,ignore_index=True)
    secondary=[]
    for (task,method,scenario),f in syn[syn.scenario.isin(["spurious","heteroscedastic"])].groupby(["task","method","scenario"]):
        a=f[(f.k==8)&(f.n==2048)&(f.imbalance==1)]
        b=f[(f.k==8)&(f.n==8192)&(f.imbalance==1)]
        j=a[["replica","gain"]].merge(b[["replica","gain"]],on="replica",suffixes=("_small","_large"),validate="one_to_one")
        secondary.append(dict(hypothesis="H3_secondary",task=task,method=method,scenario=scenario,**interval(j.gain_large-j.gain_small)))
    diagnostics=[]
    target_macro=target.groupby(["job_id","method","config"],as_index=False).primary.mean()
    source_macro=cv.groupby(["job_id","method","config"],as_index=False).primary.mean().rename(columns={"primary":"source_primary"})
    both=target_macro.merge(source_macro,on=["job_id","method","config"],validate="one_to_one")
    for (job,method),f in both.groupby(["job_id","method"]):
        chosen=int(f.loc[f.source_primary.idxmax(),"config"])
        observed=float(f[f.config==chosen].primary.iloc[0]);oracle=float(f.primary.max())
        rho=float(spearmanr(f.source_primary,f.primary).statistic) if len(f)>1 and f.source_primary.nunique()>1 and f.primary.nunique()>1 else np.nan
        diagnostics.append(dict(job_id=job,method=method,selected_config=chosen,target_best_config=int(f.loc[f.primary.idxmax(),"config"]),source_target_rank_correlation=rho,retrospective_oracle_regret=oracle-observed))
    diagnostics=pd.DataFrame(diagnostics).merge(target[["job_id",*DIMS,"model_seed"]].drop_duplicates(),on="job_id",validate="many_to_one")
    configuration_diagnostics=both.merge(target[["job_id",*DIMS,"model_seed"]].drop_duplicates(),on="job_id",validate="many_to_one").rename(columns={"primary":"target_primary"})
    return dict(selected=selected,curves=curves,units=units,macro=macro,hypotheses=hypotheses,contrasts=pd.DataFrame(contrast_rows),secondary_N=pd.DataFrame(secondary),selection_diagnostics=diagnostics,control_metrics=control_metrics,configuration_diagnostics=configuration_diagnostics)


def summaries(a):
    dimensions=DIMS[:-1]+["method"]
    effects=[]
    for key,f in a["units"].groupby(dimensions,sort=True):
        row=dict(zip(dimensions,key))
        if row["kind"]=="synthetic":
            row.update(interval(f.gain))
        else:
            row.update(replicas=len(f),mean=float(f.gain.mean()),ci_low=np.nan,ci_high=np.nan)
        row.update(min_replica=float(f.gain.min()),max_replica=float(f.gain.max()))
        effects.append(row)
    # Equal weighting: targets within a seed, seeds within source resample,
    # source resamples within a condition. None become extra target datasets.
    metrics=["primary","roc_auc","rmse","mse","mae","log_loss","balanced_accuracy","average_precision","worst_primary","worst_gain"]
    per_rep=a["macro"].groupby(DIMS+["method"],as_index=False)[metrics].mean()
    seed_counts=a["macro"].groupby(DIMS+["method"]).model_seed.nunique().rename("seed_count").reset_index()
    per_rep=per_rep.merge(seed_counts,on=DIMS+["method"],validate="one_to_one")
    per_rep=per_rep[per_rep.seed_count==2].copy()
    summary=per_rep.groupby(dimensions,as_index=False)[metrics].mean()
    summary=summary.merge(a["control_metrics"],on=dimensions,how="left",validate="one_to_one")
    return pd.DataFrame(effects),summary,per_rep


def main():
    TABLES.mkdir(parents=True,exist_ok=True)
    frozen,target,cv,timings,inventory=collect()
    a=analysis(target,cv,frozen["protocol"]);effects,summary,per_rep=summaries(a)
    outputs={"target_metrics":target,"source_cv_metrics":cv,"fit_timings":timings,"job_inventory":inventory,"selected_target_metrics":a["selected"],"all_grid_paired_effects":a["curves"],"replica_effects":a["units"],"selected_metrics_per_seed":a["macro"],"hypothesis_tests":a["hypotheses"],"hypothesis_replica_contrasts":a["contrasts"],"secondary_N_contrasts":a["secondary_N"],"selection_diagnostics":a["selection_diagnostics"],"source_target_grid_diagnostics":a["configuration_diagnostics"],"effect_summary":effects,"metric_summary":summary,"selected_metrics_per_replica":per_rep}
    for name,frame in outputs.items():
        frame.to_csv(TABLES/f"{name}.csv",index=False)
    source_records={r["context_id"]:r for r in frozen["protocol"]["source_checks"]}
    design=[];seen=set()
    for spec in frozen["protocol"]["jobs"]:
        if spec["context_id"] in seen:
            continue
        seen.add(spec["context_id"]);record=source_records[spec["context_id"]];sizes=list(record["domains"].values())
        design.append({**{k:spec[k] for k in DIMS},"context_id":spec["context_id"],"source_min_domain_rows":min(sizes),"source_max_domain_rows":max(sizes),"realized_size_ratio":max(sizes)/min(sizes),"source_hash":record["hash"]})
    pd.DataFrame(design).to_csv(TABLES/"design_contexts.csv",index=False)
    selection=a["selected"][[*DIMS,"job_id","model_seed","method","config"]].drop_duplicates()
    freq=selection.groupby(DIMS[:-1]+["method","config"]).size().rename("chosen_jobs").reset_index()
    freq["available_jobs"]=freq.groupby(DIMS[:-1]+["method"]).chosen_jobs.transform("sum")
    freq["planned_jobs"]=np.where(freq.kind=="synthetic",20,6)
    freq["missing_jobs"]=freq.planned_jobs-freq.available_jobs
    freq.to_csv(TABLES/"configuration_selection_frequency.csv",index=False)
    counts=dict(analyzed_at=now(),protocol_sha256=frozen["sha256"],jobs=len(inventory),ordinary_source_fits=int(inventory.source_fits.sum()),repeated_source_fits=int(inventory.repeated_fits.sum()),full_source_models=int(inventory.models.sum()),target_domain_metric_rows=len(target),source_validation_rows=len(cv),confirmatory_tests=len(a["hypotheses"]),significant_holm_tests=int((a["hypotheses"].p_holm<.05).sum()),files={name:len(frame) for name,frame in outputs.items()})
    counts.update(complete_jobs=int((inventory.status=="complete").sum()),partial_jobs=int((inventory.status=="complete_with_method_failure").sum()),complete_confirmatory_tests=int((a["hypotheses"].status=="complete").sum()))
    counts["analysis_code_sha256"]={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"tools/report_series_v2.py"]}
    write_json(OUT/"analysis_inventory.json",counts)
    from report_series_v2 import build_report
    build_report(a,effects,summary,per_rep,counts,frozen)
    print(json.dumps(counts,indent=2),flush=True)


if __name__=="__main__":
    main()
