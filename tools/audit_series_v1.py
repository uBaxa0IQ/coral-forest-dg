"""Artifact-first audit of v1. Metrics do NOT call nir_dg.metrics.score.

Reads only already exposed v1 targets for verification, never future holdouts.
No historical artifact is rewritten. Authoring-agent audit, not external peer review.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/audit_v1"


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def independent_metrics(task, y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise AssertionError("Nonfinite observations/predictions")
    if task == "regression":
        mse = float(np.mean((y - p) ** 2))
        denominator = np.sum((y - y.mean()) ** 2)
        return dict(mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(y-p))), r2=float(1 - np.sum((y-p)**2)/denominator))
    assert np.isin(y, [0, 1]).all() and ((p >= 0) & (p <= 1)).all()
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    auc = (rankdata(p)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    q = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    ll = -np.mean(y*np.log(q)+(1-y)*np.log1p(-q))
    ba = .5 * (np.mean(p[y == 1] >= .5) + np.mean(p[y == 0] < .5))
    order = np.argsort(-p, kind="stable")
    sorted_y, sorted_p = y[order], p[order]
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(y)-1]
    tp = np.cumsum(sorted_y)[ends]
    precision, recall = tp/(ends+1), tp/pos
    ap = np.sum(np.diff(np.r_[0., recall]) * precision)
    return dict(roc_auc=float(auc), log_loss=float(ll), balanced_accuracy=float(ba), average_precision=float(ap))


def historical_raw(name, feature_names):
    """Independent reconstruction with an explicit, exhausted v1 target allowlist."""
    if name == "acs_income":
        from folktables import ACSDataSource, ACSIncome
        ds = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=str(ROOT / "data/target_locked/acs"))
        frames = []
        for state in ["CO", "NC", "MA"]:
            raw = ds.get_data(states=[state], download=False)
            X, y, _ = ACSIncome.df_to_numpy(raw)
            f = pd.DataFrame(X, columns=ACSIncome.features)
            for c in ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]:
                f[c] = f[c].astype("Int64").astype(str)
            f["outcome"], f["environment"] = y.astype(int), state
            f["row_id"] = [f"{state}:{i}" for i in range(len(f))]
            frames.append(f)
        return pd.concat(frames, ignore_index=True)
    if name == "hospital_readmission":
        f = pd.read_csv(ROOT / "data/target_locked/hospital.csv", na_values=["?"])
        f = f.dropna(subset=["race"]).drop_duplicates("patient_nbr", keep="first")
        arrays = np.load(ROOT / "data/source/hospital_readmission/arrays.npz")
        f = f[~f.patient_nbr.astype(str).isin(arrays["entity_ids"])].copy()
        assert not set(f.patient_nbr.astype(str)) & set(arrays["entity_ids"])
        for c in ["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed"]:
            f[c] = f[c].fillna("MISSING").astype(str)
        f["outcome"], f["environment"], f["row_id"] = (f.readmitted != "NO").astype(int), "7", f.encounter_id.astype(str)
        return f.reset_index(drop=True)
    if name == "beijing_pm25":
        frames = []
        with zipfile.ZipFile(ROOT / "data/raw/beijing.zip") as outer:
            nested = [n for n in outer.namelist() if n.endswith(".zip")]
            inner = zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
            for station in ["Huairou", "Shunyi"]:
                member = next(n for n in inner.namelist() if f"_{station}_" in n and n.endswith(".csv"))
                f = pd.read_csv(inner.open(member)).dropna(subset=["PM2.5"])
                f["outcome"], f["environment"], f["row_id"] = f["PM2.5"], station, station+":"+f.No.astype(str)
                frames.append(f)
            if inner is not outer:
                inner.close()
        return pd.concat(frames, ignore_index=True)
    if name == "gas_turbine_co":
        frames = []
        with zipfile.ZipFile(ROOT / "data/raw/gas_turbine.zip") as z:
            for year in ["2014", "2015"]:
                f = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith(f"gt_{year}.csv"))))
                f["outcome"], f["environment"], f["row_id"] = f.CO, year, [f"{year}:{i}" for i in range(len(f))]
                frames.append(f)
        return pd.concat(frames, ignore_index=True)
    raise AssertionError("Audit may not read any other target")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-replay", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = ROOT / "runs/final_v1/frozen_protocol.json"
    protocol = json.loads(frozen_path.read_text())
    claimed = json.loads((ROOT / "reports/final_integrity.json").read_text())
    stored_hash = protocol.pop("protocol_hash")
    assert digest(protocol) == stored_hash
    commit = claimed["frozen_git_commit"]
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True).splitlines()
    code_paths = [p for p in paths if p.endswith(".py") and (p.startswith("src/") or p.startswith("tools/") and p.count("/") == 1)]
    frozen_blobs = {p: subprocess.check_output(["git", "show", f"{commit}:{p}"], cwd=ROOT) for p in code_paths}
    assert digest({p: sha256(b).hexdigest() for p,b in frozen_blobs.items()}) == protocol["code_hash"]
    assert all((ROOT/p).read_bytes() == b for p,b in frozen_blobs.items() if p.startswith("src/")), "Run this audit before changing v1 runtime code"
    rows = pd.read_csv(ROOT / "runs/final_v1/target_metrics.csv", dtype={"target_environment": str})
    manifest = protocol["design"]["experiment_manifest"]
    assert set(rows.experiment_id) == set(manifest)
    assert not rows.duplicated(["experiment_id", "target_environment"]).any()
    cv = pd.concat([pd.read_csv(ROOT / f"runs/study_cv_v1_{d}/source_cv_metrics.csv", dtype={"fold":str}) for d in protocol["design"]["datasets"]], ignore_index=True)
    hpo_checks=[]
    for key, e in manifest.items():
        candidates = cv[(cv.dataset == e["dataset"]) & (cv.method == e["method"]) & (cv.seed == e["seed"])]
        agg = candidates.groupby("trial").primary.mean()
        chosen = int(agg.idxmax() if e["task"] == "classification" else agg.idxmin())
        assert chosen == e["selected_trial"]
        assert json.loads(candidates[candidates.trial == chosen].parameters.iloc[0]) == e["parameters"]
        assert abs(agg.loc[chosen] - e["source_macro_score"]) < 1e-12
        hpo_checks.append(dict(experiment_id=key, selected_trial=chosen, source_score_verified=True))
    for record in protocol["design"]["hpo"]["source_cv_files"]:
        assert sha256((ROOT/record["path"]).read_bytes()).hexdigest() == record["sha256"]
    target_cache, metric_checks, prediction_checks, pair_checks, hashes = {}, [], [], [], {}
    freeze_time = datetime.fromisoformat(protocol["frozen_at"])
    for key, e in manifest.items():
        token = sha256(key.encode()).hexdigest()
        prediction_path = ROOT / "runs/final_v1/predictions" / f"{token[:16]}.parquet"
        model_path = ROOT / "runs/final_v1/models" / f"{token[:16]}.joblib"
        receipt_path = ROOT / "runs/final_v1/target_access" / f"{token}.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["experiment_id"] == key and receipt["protocol_hash"] == stored_hash
        assert datetime.fromisoformat(receipt["accessed_at"]) >= freeze_time
        pred = pd.read_parquet(prediction_path)
        pred.environment = pred.environment.astype(str)
        assert pred.row_id.is_unique
        assert set(pred.environment) == set(e["target_ids"])
        for domain, g in pred.groupby("environment"):
            metrics = independent_metrics(e["task"], g.outcome, g.prediction)
            reported = rows[(rows.experiment_id == key) & (rows.target_environment == domain)]
            assert len(reported) == 1 and len(g) == reported.num_examples.iloc[0]
            published = json.loads(reported.metrics.iloc[0])
            for metric, recomputed in metrics.items():
                delta = abs(recomputed - published[metric])
                assert delta < 1e-10 * max(1, abs(recomputed)), (key, domain, metric, delta)
                metric_checks.append(dict(experiment_id=key, domain=domain, metric=metric, recomputed=recomputed, reported=published[metric], absolute_difference=delta))
        if e["dataset"] not in target_cache:
            target_cache[e["dataset"]] = historical_raw(e["dataset"], e["feature_names"])
        raw = target_cache[e["dataset"]]
        assert np.array_equal(pred.row_id.astype(str).to_numpy(), raw.row_id.astype(str).to_numpy())
        np.testing.assert_allclose(pred.outcome, raw.outcome, atol=0, rtol=0)
        stored = joblib.load(model_path)
        indices = np.linspace(0, len(raw)-1, min(256,len(raw)), dtype=int)
        with threadpool_limits(1):
            reconstructed = stored["model"].predict_values(stored["preprocessing"].transform(raw[e["feature_names"]].iloc[indices]))
        max_error = float(np.max(np.abs(reconstructed-pred.prediction.to_numpy()[indices])))
        assert max_error < 1e-9, (key, max_error)
        prediction_checks.append(dict(experiment_id=key, target_labels_and_ids_verified=True, saved_model_replay_n=len(indices), max_prediction_difference=max_error))
        if e["paired_baseline"]:
            baseline = manifest[f"{e['dataset']}:{e['paired_baseline']}:{e['seed']}"]
            mismatch = e["parameters"].get("max_depth",4) != baseline["parameters"].get("max_depth",4) if e["method"] == "maxrm_posthoc" else False
            pair_checks.append(dict(experiment_id=key, paired_baseline=e["paired_baseline"], depth_mismatch=mismatch, method_params=json.dumps(e["parameters"]), baseline_params=json.dumps(baseline["parameters"])))
        for path in [prediction_path,model_path,receipt_path]:
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    # Recompute summaries independently from the raw per-domain metric rows.
    mean_seed = rows.groupby(["dataset","task","method","seed"]).primary.mean().reset_index()
    expected = mean_seed.groupby(["dataset","task","method"]).primary.mean()
    summary = pd.read_csv(ROOT/"reports/tables/target_summary.csv").set_index(["dataset","task","method"])
    for key,val in expected.items():
        assert abs(summary.loc[key,"target_macro_score"]-val) < 1e-10
    pd.DataFrame(metric_checks).to_csv(OUT/"metric_recalculation.csv",index=False)
    pd.DataFrame(prediction_checks).to_csv(OUT/"model_replay.csv",index=False)
    pd.DataFrame(hpo_checks).to_csv(OUT/"source_selection.csv",index=False)
    pd.DataFrame(pair_checks).to_csv(OUT/"paired_control_checks.csv",index=False)
    hashes[frozen_path.relative_to(ROOT).as_posix()]=sha256(frozen_path.read_bytes()).hexdigest()
    (OUT/"historical_artifact_hashes.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    source_replays=[]
    if args.source_replay:
        from nir_dg.datasets import load_source
        from nir_dg.protocol import source_folds
        from nir_dg.preprocessing import make_preprocessor
        from nir_dg.models import Adapter
        for name in protocol["design"]["datasets"]:
            data=load_source(name)
            fold,train,valid=next(source_folds(data))
            prep=make_preprocessor(data.X.iloc[train]); Xt=prep.fit_transform(data.X.iloc[train]); Xv=prep.transform(data.X.iloc[valid])
            selected=cv[(cv.dataset==name)&(cv.fold==fold)&(cv.seed==11)&(cv.trial==0)]
            for row in selected.itertuples():
                with threadpool_limits(1):
                    model=Adapter(row.method,data.task,11,json.loads(row.parameters)).fit(Xt,data.y[train],data.environments[train])
                    replay=independent_metrics(data.task,data.y[valid],model.predict_values(Xv))["roc_auc" if data.task=="classification" else "rmse"]
                delta=abs(replay-row.primary)
                assert delta<1e-8, (name,row.method,delta)
                source_replays.append(dict(dataset=name,fold=fold,method=row.method,seed=11,trial=0,difference=delta))
    pd.DataFrame(source_replays).to_csv(OUT/"source_fit_replay.csv",index=False)
    result=dict(audited_at=datetime.now(timezone.utc).isoformat(),independence="Independent metric implementation and artifact checks by the same authoring agent; not external peer review",protocol_digest_valid=True,frozen_code_verified_from_git=True,frozen_commit=commit,source_csv_hashes_verified=True,source_selected_configs_verified=len(hpo_checks),final_experiments=len(manifest),target_domain_rows=len(rows),independent_metric_checks=len(metric_checks),raw_target_labels_ids_verified=len(prediction_checks),saved_model_replays=len(prediction_checks),source_fit_replays=len(source_replays),maxrm_depth_mismatches=sum(x["depth_mismatch"] for x in pair_checks),historical_no_target_access_before_freeze="not independently provable from these artifacts; code and receipt chronology consistent, no OS access log",verdict="Metrics and artifact integrity verified; causal mechanism comparisons require control corrections and qualification")
    (OUT/"audit_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
