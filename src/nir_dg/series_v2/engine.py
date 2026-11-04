"""Source selection is committed per job BEFORE target generation/loading.

The manifest protects logical boundaries, not a hostile operator/OS. Raw target
files are never exposed to source CV or model adapters. Completed jobs are read
only; interrupted target jobs fail closed and require an explicit investigation.
"""
import copy
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
from nir_dg.datasets import ROOT
from nir_dg.preprocessing import make_preprocessor
from nir_dg.metrics import score
from .data import STORE,features,source_real,target_real,frame_hash
from .generators import source_frame,target_frame,seed_for
from .models import Model,grids
from .design import jobs

RUN=ROOT/"runs/expanded_v2"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    tmp.replace(path)


def runtime_hashes():
    paths=[*sorted((ROOT/"src/nir_dg").rglob("*.py")),ROOT/"tools/run_series_v2.py",ROOT/"uv.lock",ROOT/"pyproject.toml",ROOT/"configs/target_registry.json"]
    return {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}


def freeze():
    if (RUN/"frozen_protocol.json").exists():
        raise RuntimeError("Already frozen; immutable protocol")
    packages={p:importlib.metadata.version(p) for p in ["numpy","scipy","pandas","scikit-learn","pyarrow","cvxpy","clarabel","lightgbm","erasplit","anchorboosting"]}
    manifest=json.loads((STORE/"source_pool_manifest.json").read_text())
    specs=jobs()
    source_checks=[]
    seen=set()
    for s in specs:
        if s["context_id"] in seen:
            continue
        seen.add(s["context_id"])
        f=source_frame(s) if s["kind"]=="synthetic" else source_real(s)
        source_checks.append({"context_id":s["context_id"],"hash":frame_hash(f),"rows":len(f),"domains":f.groupby("__env").size().to_dict()})
    body=dict(series="expanded_v2",frozen_at=now(),runtime_hashes=runtime_hashes(),packages=packages,python=platform.python_version(),platform=platform.platform(),source_pools=manifest,source_checks=source_checks,jobs=specs,grids={task:grids(task) for task in ["classification","regression"]},target_n_per_synthetic_domain=2000,source_validation="two predeclared leave-one-source-domain folds; temporal rolling with 24h embargo",selection="highest macro heldout-domain AUROC / lowest macro normalized MSE; grid order breaks ties",diagnostic_curves="every registered grid point, no target-based selection",replica_unit="10 independent synthetic draws; real replicates are overlapping source resamples",primary_hypothesis_families={"H1":["K16-K4 benefit difference, spurious, N8192, balanced; nine method-task tests"],"H2":["8:1-minus-balanced benefit difference, spurious, K8,N8192; nine method-task tests"]},replay="all methods/configurations of contexts with replica=0, seed=11 on first source fold; max prediction abs diff <=1e-8",target_policy="new registered targets only; no outcome-dependent changes or retries")
    body["statistics"]={"classification_effect":"method AUROC minus own config-0 control, macro target-domain then seed means","regression_effect":"(control MSE minus method MSE)/full-source variance, macro target-domain then seed means","primary_methods":{"classification":["irf","crf","meta","anchor"],"regression":["irf","maxrm","era","directional_era","anchor"]},"primary_testing":"two-sided exact sign test on independent replica contrasts; zeros omitted; Holm across nine tests separately for H1 and H2","intervals":"two-sided Student t 95% intervals over ten independent synthetic data replicas; normality approximation, sign test does not assume normality","secondary":"H3/H4/H5/H6/H7 and real datasets descriptive; no confirmatory claims from unadjusted secondary p-values"}
    body["source_code_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    body["source_holdouts"]={"acs":"first/last in original selected source-state order: CA and AZ for both K","synthetic_and_beijing":"first and last ordered selected source domains","appliances":["2016-W09","2016-W12"],"temporal_embargo_hours":24}
    body["research_documents_sha256"]={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in ["docs/series_v2/research_plan.md","docs/series_v2/preregistration.md","reports/audit_v1/audit_results.json","reports/series_v2/tests_pre_freeze.xml"]}
    write_json(RUN/"frozen_protocol.json",{"protocol":body,"sha256":digest(body)})
    return body


def verify_freeze():
    f=json.loads((RUN/"frozen_protocol.json").read_text(encoding="utf-8"))
    if digest(f["protocol"])!=f["sha256"] or runtime_hashes()!=f["protocol"]["runtime_hashes"]:
        raise RuntimeError("Frozen protocol/runtime mismatch")
    if any(importlib.metadata.version(p)!=version for p,version in f["protocol"]["packages"].items()):
        raise RuntimeError("Frozen package version mismatch")
    for name,meta in f["protocol"]["source_pools"].items():
        if sha256((STORE/f"pool_{name}.parquet").read_bytes()).hexdigest()!=meta["file_sha256"]:
            raise RuntimeError("Frozen source pool mismatch")
    return f


def folds(frame,spec):
    domains=list(dict.fromkeys(frame.__env)) if spec["dataset"].startswith("acs_") else sorted(frame.__env.unique())
    if spec["dataset"]=="appliances":
        for heldout in ["2016-W09","2016-W12"]:
            index=domains.index(heldout)
            valid=np.flatnonzero(frame.__env==domains[index])
            start=int(frame.iloc[valid].__time.min())
            train=np.flatnonzero((frame.__env.isin(domains[:index])) & (frame.__time<start-24*3600*10**9))
            if not len(train) or not len(valid):
                raise RuntimeError("Empty temporal fold")
            assert frame.iloc[train].__time.max()<frame.iloc[valid].__time.min()-24*3600*10**9
            yield train,valid
    else:
        for domain in [domains[0],domains[-1]]:
            yield np.flatnonzero(frame.__env!=domain),np.flatnonzero(frame.__env==domain)


def pack_folds(frame,spec):
    X=frame[features(frame)]
    result=[]
    for tr,va in folds(frame,spec):
        processor=make_preprocessor(X.iloc[tr]);a=processor.fit_transform(X.iloc[tr]);b=processor.transform(X.iloc[va])
        result.append((tr,va,a,b))
    return result


def primary(task,y,p,scale):
    return score(task,y,p)["roc_auc"] if task=="classification" else -float(np.mean(((y-p)/scale)**2))


def source_stage(spec,save=True,replay=True):
    started=time.monotonic()
    frame=source_frame(spec) if spec["kind"]=="synthetic" else source_real(spec)
    y=frame.__y.to_numpy();env=frame.__env.to_numpy();task=spec["task"]
    scale=max(float(y.std()),1e-8)
    cv=[];source_predictions=[];replays=[];models={};times=[]
    packed=pack_folds(frame,spec)
    for method,grid in grids(task).items():
        for ci,params in enumerate(grid):
            for fi,(tr,va,a,b) in enumerate(packed):
                tick=time.monotonic()
                m=Model(method,task,spec["model_seed"],params).fit(a,y[tr],env[tr]);p=m.predict(b)
                duration=time.monotonic()-tick
                cv.append(dict(method=method,config=ci,fold=fi,primary=primary(task,y[va],p,scale),**score(task,y[va],p)))
                source_predictions.append(pd.DataFrame({"method":method,"config":ci,"fold":fi,"row_id":frame.iloc[va].__row_id.to_numpy(),"y":y[va],"prediction":p}))
                times.append(dict(method=method,config=ci,fold=fi,seconds=duration))
                if replay and fi==0 and spec["replica"]==0 and spec["model_seed"]==11:
                    probe=a[:256];again=Model(method,task,spec["model_seed"],params).fit(a,y[tr],env[tr])
                    diff=float(np.max(np.abs(m.predict(probe)-again.predict(probe))))
                    if diff>1e-8:
                        raise RuntimeError(f"Reproducibility failure {method}: {diff}")
                    replays.append(dict(method=method,config=ci,max_abs_prediction_difference=diff))
    table=pd.DataFrame(cv)
    selected={}
    for method,grid in grids(task).items():
        means=table[table.method==method].groupby("config").primary.mean()
        selected[method]=int(means.idxmax())
    processor=make_preprocessor(frame[features(frame)])
    full=processor.fit_transform(frame[features(frame)])
    for method,grid in grids(task).items():
        for ci,params in enumerate(grid):
            tick=time.monotonic()
            m=Model(method,task,spec["model_seed"],params).fit(full,y,env)
            models[f"{method}:{ci}"]=m
            times.append(dict(method=method,config=ci,fold="full",seconds=time.monotonic()-tick))
    result=dict(spec=spec,source_hash=frame_hash(frame),source_rows=len(frame),source_domains=frame.groupby("__env").size().to_dict(),source_y_scale=scale,selected=selected,cv=cv,replays=replays,times=times,source_completed_at=now(),source_seconds=time.monotonic()-started)
    if save:
        path=RUN/"jobs"/spec["job_id"];path.mkdir(parents=True,exist_ok=True)
        (RUN/"models").mkdir(exist_ok=True)
        model_path=RUN/"models"/(spec["job_id"]+".joblib")
        joblib.dump((processor,models),model_path,compress=3)
        result["model_sha256"]=sha256(model_path.read_bytes()).hexdigest()
        (RUN/"predictions").mkdir(exist_ok=True)
        pd.concat(source_predictions,ignore_index=True).to_parquet(RUN/"predictions"/(spec["job_id"]+"_source.parquet"),index=False)
        write_json(path/"source_selection.json",result)
    return result,(processor,models)


def run_job(spec):
    frozen=verify_freeze(); path=RUN/"jobs"/spec["job_id"]
    if (path/"result.json").exists():
        return {"job_id":spec["job_id"],"status":"already_complete"}
    if (path/"target_access.json").exists():
        raise RuntimeError("Interrupted after target access: fail closed, no silent retry")
    if spec not in frozen["protocol"]["jobs"]:
        raise RuntimeError("Unregistered job")
    selection_path=path/"source_selection.json"
    if selection_path.exists():
        selected=json.loads(selection_path.read_text(encoding="utf-8"))
        model_path=RUN/"models"/(spec["job_id"]+".joblib")
        if sha256(model_path.read_bytes()).hexdigest()!=selected["model_sha256"]:
            raise RuntimeError("Saved pretarget model modified")
        processor,models=joblib.load(model_path)
    else:
        selected,(processor,models)=source_stage(spec)
    expected=next(s["hash"] for s in frozen["protocol"]["source_checks"] if s["context_id"]==spec["context_id"])
    if selected["source_hash"]!=expected:
        raise RuntimeError("Source fingerprint mismatch")
    # The receipt is durably written BEFORE requesting any target value.
    write_json(path/"target_access.json",dict(authorized_at=now(),protocol_sha256=frozen["sha256"],source_selection_sha256=sha256(selection_path.read_bytes()).hexdigest(),model_sha256=selected["model_sha256"],job_id=spec["job_id"]))
    target=target_frame(spec,frozen["protocol"]["target_n_per_synthetic_domain"],frozen["sha256"]) if spec["kind"]=="synthetic" else target_real(spec,frozen["sha256"])
    columns=features(target)
    transformed=processor.transform(target[columns]);y=target.__y.to_numpy();env=target.__env.to_numpy()
    source=source_frame(spec) if spec["kind"]=="synthetic" else source_real(spec)
    if set(source.__row_id)&set(target.__row_id):
        raise RuntimeError("Source/target row overlap")
    rows=[];preds=[]
    for key,model in models.items():
        method,ci=key.split(":");ci=int(ci);p=model.predict(transformed)
        for e in np.unique(env):
            mask=env==e
            rows.append(dict(method=method,config=ci,selected=ci==selected["selected"][method],target_domain=str(e),n=int(mask.sum()),**score(spec["task"],y[mask],p[mask])))
        preds.append(pd.DataFrame(dict(method=method,config=ci,row_id=target.__row_id.to_numpy(),environment=env,y=y,prediction=p)))
    pd.concat(preds,ignore_index=True).to_parquet(RUN/"predictions"/(spec["job_id"]+"_target.parquet"),index=False)
    result=dict(spec=spec,protocol_sha256=frozen["sha256"],source_hash=selected["source_hash"],target_hash=frame_hash(target),source_y_scale=selected["source_y_scale"],source_selected=selected["selected"],metrics=rows,completed_at=now(),status="complete")
    write_json(path/"result.json",result)
    return {"job_id":spec["job_id"],"status":"complete","source_seconds":selected["source_seconds"]}
