"""Independent equations over stored predictions; no model selection or fitting.

This checker is deliberately outside the frozen training runtime. Its metric
implementation is the separately written rank/NumPy implementation from v1.
"""
import os
import argparse
for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[name]="1"
from datetime import datetime
from hashlib import sha256
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_series_v1 import independent_metrics
from nir_dg.series_v2.engine import RUN,ROOT,verify_freeze,digest,write_json,now,folds
from nir_dg.series_v2.data import STORE,source_real,frame_hash,features,target_real
from nir_dg.series_v2.generators import source_frame,target_frame
from nir_dg.series_v2.models import grids

OUT=ROOT/"reports/series_v2/audit"


def forest_signature(model):
    h=sha256()
    for tree in model.estimators_:
        for a in [tree.tree_.children_left,tree.tree_.children_right,tree.tree_.feature,tree.tree_.threshold,tree.tree_.value]:
            h.update(a.tobytes())
    return h.hexdigest()


def verify_pair_models(models,task):
    checks=[]
    if task=="regression" and "maxrm:0" in models:
        a,b=models["maxrm:0"].model_,models["maxrm:1"].model_
        for ta,tb,ia,ib in zip(a.members_,b.members_,a.bootstrap_indices_,b.bootstrap_indices_):
            assert np.array_equal(ia,ib)
            for name in ["children_left","children_right","feature","threshold"]:
                assert np.array_equal(getattr(ta.tree_,name),getattr(tb.tree_,name))
        assert all(v["after"]<=v["before"]+1e-5*max(1.,v["before"]) for v in b.optimization_)
        checks.append("maxrm_identical_structure_bootstrap_and_feasible_risk")
    elif task=="classification":
        for family in ["meta","meta_masked"]:
            base=models[f"{family}:0"].model_
            for ci in [1,2]:
                other=models[f"{family}:{ci}"].model_
                assert len(base.members_)==len(other.members_)
                for (fa,xa),(fb,xb) in zip(base.members_,other.members_):
                    assert np.array_equal(xa,xb) and forest_signature(fa)==forest_signature(fb)
            checks.append(f"{family}_identical_trained_members")
    for family in ["irf"]+(["crf"] if task=="classification" else []):
        base=models[f"{family}:0"].model_
        for ci in [1,2]:
            other=models[f"{family}:{ci}"].model_
            assert all(np.array_equal(a,b) for a,b in zip(base.bootstrap_indices_,other.bootstrap_indices_))
        checks.append(f"{family}_identical_bootstrap_and_capacity")
    return checks


def numeric_tree_audit(models,processor,source,task,job):
    X=processor.transform(source[features(source)]);env=source.__env.to_numpy();rows=[]
    for key,m in models.items():
        if not key.startswith(("irf:","crf:")):
            continue
        bad_leaf=0;nonfinite=0;support=0;empty=0
        def visit(node,ids):
            nonlocal bad_leaf,nonfinite,support,empty
            nonfinite+=int(not np.isfinite(node.value));empty+=int(len(ids)==0)
            if node.feature<0:
                bad_leaf+=int(len(ids)<m.model_.min_samples_leaf)
                return
            left=X[ids,node.feature]<=node.threshold
            if task=="regression" and key.startswith("irf:"):
                active=set(env[ids])
                support+=int(set(env[ids[left]])!=active or set(env[ids[~left]])!=active)
            visit(node.left,ids[left]);visit(node.right,ids[~left])
        for tree,ids in zip(m.model_.trees_,m.model_.bootstrap_indices_):
            visit(tree,ids)
        rows.append(dict(job_id=job,model=key,leaves_below_registered_minimum=bad_leaf,empty_nodes=empty,nonfinite_nodes=nonfinite,regression_support_violations=support))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--partial-limit",type=int)
    args=parser.parse_args()
    global OUT
    if args.partial_limit:
        OUT=OUT/"partial"
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=verify_freeze();protocol=frozen["protocol"]
    for path,h in protocol["research_documents_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest()==h
    registered=protocol["jobs"]
    addendum=json.loads((RUN/"source_failure_addendum.json").read_text(encoding="utf-8"))
    assert digest(addendum["addendum"])==addendum["sha256"]
    assert sha256((ROOT/"tools/recover_registered_v2.py").read_bytes()).hexdigest()==addendum["addendum"]["driver_sha256"]
    recovery_jobs=set(addendum["addendum"]["jobs"])
    replay_provenance=json.loads((OUT/"replay_input_provenance.json").read_text(encoding="utf-8")) if not args.partial_limit else {"jobs":{}}
    # Explicit forensic input reconstruction; no recorded prediction/metric changes.
    available={p.parent.name for p in (RUN/"jobs").glob("*/result.json")}
    if args.partial_limit:
        registered=[s for s in registered if s["job_id"] in available][:args.partial_limit]
    else:
        assert available=={s["job_id"] for s in registered},"Incomplete or unregistered jobs"
    metric_checks=[];selection_checks=[];model_checks=[];ledger=[];hashes={};target_cache={};numeric_checks=[]
    expected_sources={s["context_id"]:s["hash"] for s in protocol["source_checks"]}
    runtime_replays=0
    for number,spec in enumerate(registered,1):
        job=spec["job_id"];path=RUN/"jobs"/job
        paths={name:path/f"{name}.json" for name in ["result","source_selection","target_access"]}
        r,sel,receipt=[json.loads(paths[name].read_text(encoding="utf-8")) for name in ["result","source_selection","target_access"]]
        omitted=["maxrm"] if job in recovery_jobs else []
        assert r["spec"]==spec and sel["spec"]==spec
        assert r["status"]==("complete_with_method_failure" if omitted else "complete")
        assert r.get("omitted_methods",[])==sel.get("omitted_methods",[])==omitted
        if omitted:
            assert r["execution_addendum_sha256"]==sel["execution_addendum_sha256"]==addendum["sha256"]
            assert sel["recovery_driver_sha256"]==addendum["addendum"]["driver_sha256"]
            assert datetime.fromisoformat(addendum["addendum"]["recorded_at"])<datetime.fromisoformat(sel["source_completed_at"])
        assert r["protocol_sha256"]==receipt["protocol_sha256"]==frozen["sha256"]
        assert receipt["source_selection_sha256"]==sha256(paths["source_selection"].read_bytes()).hexdigest()
        assert receipt["model_sha256"]==sel["model_sha256"]
        assert r["source_hash"]==sel["source_hash"]==expected_sources[spec["context_id"]]
        assert datetime.fromisoformat(protocol["frozen_at"])<datetime.fromisoformat(sel["source_completed_at"])<=datetime.fromisoformat(receipt["authorized_at"])<datetime.fromisoformat(r["completed_at"])
        source=source_frame(spec) if spec["kind"]=="synthetic" else source_real(spec)
        assert frame_hash(source)==sel["source_hash"]
        key=(spec["kind"],spec["dataset"],spec["task"],spec["scenario"],spec["replica"] if spec["kind"]=="synthetic" else 0)
        if key not in target_cache:
            target_cache[key]=target_frame(spec,protocol["target_n_per_synthetic_domain"],frozen["sha256"]) if spec["kind"]=="synthetic" else target_real(spec,frozen["sha256"])
        target=target_cache[key]
        assert frame_hash(target)==r["target_hash"]
        assert not set(source.__row_id)&set(target.__row_id)
        assert source.__row_id.is_unique and target.__row_id.is_unique
        assert set(features(source))==set(features(target))
        sp=RUN/"predictions"/f"{job}_source.parquet";tp=RUN/"predictions"/f"{job}_target.parquet"
        raw_source=pd.read_parquet(sp);raw_target=pd.read_parquet(tp)
        cv=pd.DataFrame(sel["cv"])
        expected_methods=set(protocol["grids"][spec["task"]])-set(omitted)
        assert set(sel["selected"])==set(r["source_selected"])==set(cv.method)==expected_methods
        assert not cv.duplicated(["method","config","fold"]).any()
        assert len(cv)==2*sum(len(protocol["grids"][spec["task"]][m]) for m in expected_methods)
        source_truth=source.set_index("__row_id")
        expected_validation={fi:source.iloc[va].__row_id.to_list() for fi,(_,va) in enumerate(folds(source,spec))}
        for (method,ci,fold),g in raw_source.groupby(["method","config","fold"],sort=False):
            reference=cv[(cv.method==method)&(cv.config==ci)&(cv.fold==fold)].iloc[0]
            assert g.row_id.to_list()==expected_validation[int(fold)]
            expected_y=source_truth.loc[g.row_id,"__y"].to_numpy()
            assert np.array_equal(expected_y,g.y.to_numpy())
            values=independent_metrics(spec["task"],g.y,g.prediction)
            independently_primary=values["roc_auc"] if spec["task"]=="classification" else -values["mse"]/sel["source_y_scale"]**2
            assert abs(independently_primary-reference.primary)<=1e-10
            for name,value in values.items():
                delta=abs(value-reference[name]);assert delta<=1e-9+1e-11*abs(value)
                metric_checks.append(dict(job_id=job,stage="source",method=method,config=int(ci),fold_or_domain=str(fold),metric=name,absolute_difference=delta))
        for method,grid in grids(spec["task"]).items():
            if method in omitted:
                assert method not in set(cv.method) and method not in r["source_selected"]
                continue
            chosen=int(cv[cv.method==method].groupby("config").primary.mean().idxmax())
            assert chosen==sel["selected"][method]==r["source_selected"][method]
            selection_checks.append(dict(job_id=job,method=method,config=chosen))
        metrics=pd.DataFrame(r["metrics"])
        assert set(metrics.method)==expected_methods
        assert not metrics.duplicated(["method","config","target_domain"]).any()
        assert len(metrics)==target.__env.nunique()*sum(len(protocol["grids"][spec["task"]][m]) for m in expected_methods)
        assert all(bool(row.selected)==(int(row.config)==sel["selected"][row.method]) for row in metrics.itertuples())
        full_predictions={}
        truth=target.set_index("__row_id")
        for (method,ci),g in raw_target.groupby(["method","config"],sort=False):
            assert len(g)==len(target) and g.row_id.is_unique and set(g.row_id)==set(target.__row_id)
            expected=truth.loc[g.row_id]
            assert np.array_equal(expected.__y.to_numpy(),g.y.to_numpy())
            assert np.array_equal(expected.__env.to_numpy(),g.environment.to_numpy())
            full_predictions[(method,int(ci))]=g.prediction.to_numpy()
            for env,sub in g.groupby("environment",sort=False):
                reference=metrics[(metrics.method==method)&(metrics.config==ci)&(metrics.target_domain==env)].iloc[0]
                assert reference.n==len(sub)
                for name,value in independent_metrics(spec["task"],sub.y,sub.prediction).items():
                    delta=abs(value-reference[name]);assert delta<=1e-9+1e-11*abs(value)
                    metric_checks.append(dict(job_id=job,stage="target",method=method,config=int(ci),fold_or_domain=str(env),metric=name,absolute_difference=delta))
        model_path=RUN/"models"/f"{job}.joblib"
        assert sha256(model_path.read_bytes()).hexdigest()==sel["model_sha256"]
        processor,models=joblib.load(model_path)
        expected_keys={f"{m}:{ci}" for m,g in protocol["grids"][spec["task"]].items() if m not in omitted for ci in range(len(g))}
        assert set(models)==expected_keys
        checks=verify_pair_models(models,spec["task"])
        numeric_checks+=numeric_tree_audit(models,processor,source,spec["task"],job)
        ids=np.unique(np.linspace(0,len(target)-1,min(256,len(target))).astype(int))
        probe_frame=target.iloc[ids][features(target)].copy()
        input_mode="cached_frame"
        if job in replay_provenance["jobs"]:
            note=replay_provenance["jobs"][job]
            assert spec["dataset"]=="beijing_pm25" and int(target.wd.isna().sum())==note["missing_rows"]
            probe_frame.loc[probe_frame.wd.isna(),"wd"]=np.nan
            input_mode="reconstructed_original_csv_nan_wd"
        probe=processor.transform(probe_frame)
        for model_key,model in models.items():
            method,ci=model_key.split(":")
            pred=model.predict(probe)
            difference=float(np.max(np.abs(pred-full_predictions[(method,int(ci))][ids])))
            assert difference<=1e-8,(job,model_key,difference,input_mode)
            model_checks.append(dict(job_id=job,method=method,config=int(ci),probe_rows=len(ids),max_abs_difference=difference,input_mode=input_mode))
        if spec["task"]=="regression":
            assert np.max(np.abs(full_predictions[("era",0)]-full_predictions[("directional_era",0)]))<=1e-12
        runtime_replays+=len(sel["replays"])
        ledger.append(dict(job_id=job,source_hash=sel["source_hash"],target_hash=r["target_hash"],pair_checks=";".join(checks),source_completed_at=sel["source_completed_at"],target_authorized_at=receipt["authorized_at"],completed_at=r["completed_at"]))
        for p in [*paths.values(),sp,tp,model_path]:
            hashes[p.relative_to(ROOT).as_posix()]=sha256(p.read_bytes()).hexdigest()
        if number%10==0:
            print(f"AUDIT {number}/{len(registered)}",flush=True)
        del raw_source,raw_target,models,processor,full_predictions
    pd.DataFrame(metric_checks).to_csv(OUT/"independent_metric_checks.csv",index=False)
    pd.DataFrame(model_checks).to_csv(OUT/"model_prediction_replays.csv",index=False)
    pd.DataFrame(selection_checks).to_csv(OUT/"source_selection_checks.csv",index=False)
    pd.DataFrame(ledger).to_csv(OUT/"access_and_pair_ledger.csv",index=False)
    numeric_frame=pd.DataFrame(numeric_checks)
    numeric_frame.to_csv(OUT/"tree_numeric_checks.csv",index=False)
    write_json(OUT/"artifact_hashes.json",hashes)
    result=dict(completed_at=now(),protocol_sha256=frozen["sha256"],registered_jobs=len(protocol["jobs"]),audited_jobs=len(registered),completed_jobs_at_start=len(available),source_selection_checks=len(selection_checks),independent_metric_checks=len(metric_checks),saved_model_replays=len(model_checks),repeated_source_fits=runtime_replays,all_audited_pairs_checked=True,no_source_target_row_overlap=True,max_metric_absolute_difference=max(x["absolute_difference"] for x in metric_checks),max_model_prediction_difference=max(x["max_abs_difference"] for x in model_checks),historical_os_no_access_proof=False,checker_independence="Separate equations/checker, same authoring agent; not external peer review",checker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),status="passed_partial" if args.partial_limit else "passed")
    result["numeric_tree_findings"]={c:int(numeric_frame[c].sum()) for c in ["leaves_below_registered_minimum","empty_nodes","nonfinite_nodes","regression_support_violations"]}
    result["source_failed_jobs"]=len(recovery_jobs)
    result["complete_jobs"]=len(protocol["jobs"])-len(recovery_jobs)
    result["salvaged_jobs_with_unavailable_maxrm"]=len(recovery_jobs)
    result["execution_addendum_sha256"]=addendum["sha256"]
    if any(result["numeric_tree_findings"].values()) and not args.partial_limit:
        result["status"]="passed_with_numeric_qualifications"
    result["metric_equations_sha256"]=sha256((ROOT/"tools/audit_series_v1.py").read_bytes()).hexdigest()
    result["forensic_input_reconstruction_jobs"]=list(replay_provenance["jobs"])
    result["replay_input_provenance_sha256"]=sha256((OUT/"replay_input_provenance.json").read_bytes()).hexdigest() if replay_provenance["jobs"] else None
    if replay_provenance["jobs"] and not args.partial_limit:
        result["status"]="passed_with_input_and_numeric_qualifications"
    write_json(OUT/"final_integrity.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
