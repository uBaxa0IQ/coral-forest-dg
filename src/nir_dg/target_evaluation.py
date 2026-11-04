"""Protected, preregistered final evaluation. Source selection lives elsewhere."""
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
from time import perf_counter
import zipfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nir_dg.datasets import ROOT, ACS_TARGET, BEIJING_TARGET, load_source
from nir_dg.experiment import append_record
from nir_dg.metrics import score
from nir_dg.models import Adapter, METHODS
from nir_dg.preprocessing import make_preprocessor
from nir_dg.protocol import ProtocolError, TargetVault, freeze_protocol, source_folds, verified_protocol

TARGET_IDS = {"acs_income": ACS_TARGET, "hospital_readmission": ["7"], "beijing_pm25": BEIJING_TARGET, "gas_turbine_co": ["2014", "2015"]}


@dataclass(frozen=True)
class TargetData:
    X: pd.DataFrame
    y: np.ndarray
    environments: np.ndarray
    row_ids: np.ndarray


def _load_preregistered_target(entry):
    name = entry["dataset"]
    if entry["target_ids"] != TARGET_IDS[name]:
        raise ProtocolError("Target identifiers do not match the frozen loader policy")
    if name == "acs_income":
        from folktables import ACSDataSource, ACSIncome
        source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=str(ROOT / "data/target_locked/acs"))
        frames = []
        for state in entry["target_ids"]:
            raw = source.get_data(states=[state], download=True)
            X, y, _ = ACSIncome.df_to_numpy(raw)
            frame = pd.DataFrame(X, columns=ACSIncome.features)
            for col in ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]:
                frame[col] = frame[col].astype("Int64").astype(str)
            frame["y"], frame["env"] = y.astype(int), state
            frame["row_id"] = [f"{state}:{i}" for i in range(len(frame))]
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
    elif name == "hospital_readmission":
        frame = pd.read_csv(ROOT / "data/target_locked/hospital.csv", na_values=["?"])
        frame = frame.dropna(subset=["race"]).drop_duplicates("patient_nbr", keep="first")
        source = load_source(name)
        # Prespecified evaluation exclusion: never remove source rows after seeing target.
        frame = frame[~frame.patient_nbr.astype(str).isin(source.entity_ids)].copy()
        frame["y"] = (frame.readmitted != "NO").astype(int)
        frame["env"], frame["row_id"] = "7", frame.encounter_id.astype(str)
        for col in ["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed"]:
            frame[col] = frame[col].fillna("MISSING").astype(str)
    elif name == "beijing_pm25":
        frames = []
        with zipfile.ZipFile(ROOT / "data/raw/beijing.zip") as outer:
            nested = [n for n in outer.namelist() if n.endswith(".zip")]
            inner = zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
            try:
                for station in entry["target_ids"]:
                    path = next(n for n in inner.namelist() if n.endswith(".csv") and f"_{station}_" in n)
                    frame = pd.read_csv(inner.open(path)).dropna(subset=["PM2.5"])
                    frame["y"], frame["env"] = frame["PM2.5"], station
                    frame["row_id"] = station + ":" + frame.No.astype(str)
                    frames.append(frame)
            finally:
                if inner is not outer:
                    inner.close()
        frame = pd.concat(frames, ignore_index=True)
    elif name == "gas_turbine_co":
        frames = []
        with zipfile.ZipFile(ROOT / "data/raw/gas_turbine.zip") as z:
            for year in entry["target_ids"]:
                path = next(n for n in z.namelist() if n.endswith(f"gt_{year}.csv"))
                frame = pd.read_csv(z.open(path))
                frame["y"], frame["env"] = frame.CO, year
                frame["row_id"] = [f"{year}:{i}" for i in range(len(frame))]
                frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
    else:
        raise ProtocolError("No target loader registered")
    if not set(frame.env.unique()).issubset(entry["target_ids"]):
        raise ProtocolError("Unexpected target environment")
    return TargetData(frame[entry["feature_names"]].copy(), frame.y.to_numpy(), frame.env.to_numpy(), frame.row_id.to_numpy())


def select_source_configs(cv_paths, expected_datasets):
    """Require complete finite source folds for each candidate; no target argument."""
    frames = [pd.read_csv(path) for path in cv_paths]
    all_rows = pd.concat(frames, ignore_index=True)
    if set(all_rows.dataset.unique()) != set(expected_datasets):
        raise ProtocolError("Unexpected or missing source-CV datasets")
    selections = []
    for (name, seed, method), rows in all_rows.groupby(["dataset", "seed", "method"], sort=True):
        task = rows.task.iloc[0]
        data = load_source(name)
        expected_folds = {e for e, _, _ in source_folds(data)}
        candidates = []
        for trial, trials in rows.groupby("trial"):
            if set(trials.fold.astype(str)) != expected_folds or len(trials) != len(expected_folds):
                continue
            if not trials.outcome.eq("ok").all() or not np.isfinite(trials.primary).all():
                continue
            candidates.append((float(trials.primary.mean()), int(trial), json.loads(trials.parameters.iloc[0])))
        if not candidates:
            raise ProtocolError(f"No complete finite source-validation candidate for {name}/{seed}/{method}")
        candidates.sort(key=lambda x: ((-x[0]) if task == "classification" else x[0], x[1]))
        mean, trial, params = candidates[0]
        selections.append(dict(dataset=name, seed=int(seed), method=method, parameters=params, source_macro_score=mean, selected_trial=trial, num_source_folds=len(expected_folds)))
    return selections


def preregister(cv_paths, run_dir):
    run_dir = Path(run_dir)
    if (run_dir / "frozen_protocol.json").exists():
        raise ProtocolError("Already registered")
    datasets = list(TARGET_IDS)
    choices = select_source_configs(cv_paths, datasets)
    manifest = {}
    for choice in choices:
        name, method, seed = choice["dataset"], choice["method"], choice["seed"]
        data = load_source(name)
        identifier = f"{name}:{method}:{seed}"
        manifest[identifier] = dict(choice, task=data.task, feature_names=list(data.X), target_ids=TARGET_IDS[name], implementation_status=METHODS[method].fidelity, paired_baseline=METHODS[method].paired_baseline)
    expected = {(name, method, seed) for name in datasets for method in __import__("nir_dg.models", fromlist=["applicable_methods"]).applicable_methods(load_source(name).task) for seed in [11, 23, 47]}
    actual = {(e["dataset"], e["method"], e["seed"]) for e in manifest.values()}
    if actual != expected:
        raise ProtocolError("Incomplete experiment manifest")
    design = dict(datasets=datasets, methods=sorted({e["method"] for e in manifest.values()}), seeds=[11, 23, 47],
        metrics={"classification": "roc_auc", "regression": "rmse", "aggregation": "macro_per_target_domain", "threshold": .5},
        hpo={"selection": "source CV completed; selected per seed; no target used", "source_cv_files": [{"path": str(Path(p).relative_to(ROOT)), "sha256": sha256(Path(p).read_bytes()).hexdigest()} for p in cv_paths]},
        source_folds={name: [fold for fold, _, _ in source_folds(load_source(name))] for name in datasets},
        source_fingerprints={name: load_source(name).fingerprint() for name in datasets},
        target_ids=TARGET_IDS, experiment_manifest=manifest,
        target_policy="Full eligible target domains; drop missing outcomes; hospital drop missing race, first encounter per patient and exclude source-training patients; no target subsampling or fitting", 
        limitations=["Exploratory capped source samples of up to 400/domain", "3 seeds do not represent independent domains", "Meta-forests is a labelled independent interpretation", "CRF is paper-objective reimplementation, not author R port", "Paired baseline HPO grids can be smaller; actual costs reported", "No claim of reproduction of published numerical tables"])
    value = freeze_protocol(run_dir, design, ROOT)
    pd.DataFrame([{**e, "parameters": json.dumps(e["parameters"], sort_keys=True), "feature_names": json.dumps(e["feature_names"]), "target_ids": json.dumps(e["target_ids"]), "experiment_id": identifier} for identifier, e in manifest.items()]).to_csv(run_dir / "experiment_manifest.csv", index=False)
    (run_dir / "selected_source_configs.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    return value


def evaluate_registered(run_dir, datasets=None):
    run_dir = Path(run_dir)
    protocol = verified_protocol(run_dir, ROOT)
    metrics_path = run_dir / "target_metrics.csv"
    done = set()
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        for identifier, rows in old.groupby("experiment_id"):
            expected_ids = protocol["design"]["experiment_manifest"][identifier]["target_ids"]
            if len(rows) != len(expected_ids) or set(rows.target_environment.astype(str)) != set(expected_ids):
                raise ProtocolError("Partial target metrics require a documented recovery; refusing silent skip")
            done.add(identifier)
    vault = TargetVault(run_dir, ROOT, _load_preregistered_target)
    for identifier, entry in protocol["design"]["experiment_manifest"].items():
        if identifier in done or (datasets is not None and entry["dataset"] not in datasets):
            continue
        data = load_source(entry["dataset"])
        if data.fingerprint() != protocol["design"]["source_fingerprints"][data.name]:
            raise ProtocolError("Source changed after freeze")
        preprocessor = make_preprocessor(data.X)
        X = preprocessor.fit_transform(data.X)
        start = perf_counter()
        with threadpool_limits(limits=1):
            model = Adapter(entry["method"], data.task, entry["seed"], entry["parameters"]).fit(X, data.y, data.environments)
            fit_seconds = perf_counter() - start
            key = sha256(identifier.encode()).hexdigest()[:16]
            (run_dir / "models").mkdir(exist_ok=True)
            joblib.dump(dict(model=model, preprocessing=preprocessor, protocol_hash=protocol["protocol_hash"]), run_dir / "models" / f"{key}.joblib")
            # Last operation before reading target is verification of frozen design/code.
            target = vault.load_for_evaluation(identifier)
            start = perf_counter()
            prediction = model.predict_values(preprocessor.transform(target.X))
            predict_seconds = perf_counter() - start
        (run_dir / "predictions").mkdir(exist_ok=True)
        pd.DataFrame(dict(row_id=target.row_ids, environment=target.environments, outcome=target.y, prediction=prediction)).to_parquet(run_dir / "predictions" / f"{key}.parquet", index=False)
        for environment in entry["target_ids"]:
            mask = target.environments == environment
            metrics = score(data.task, target.y[mask], prediction[mask])
            append_record(metrics_path, dict(experiment_id=identifier, dataset=data.name, task=data.task, method=entry["method"], seed=entry["seed"], target_environment=environment, primary=metrics["roc_auc" if data.task == "classification" else "rmse"], metrics=json.dumps(metrics, sort_keys=True), num_examples=int(mask.sum()), fit_seconds=fit_seconds, predict_seconds=predict_seconds, implementation_status=entry["implementation_status"], paired_baseline=entry["paired_baseline"], frozen_protocol_hash=protocol["protocol_hash"], evaluated_at=datetime.now(timezone.utc).isoformat()))
        print(identifier, "target_evaluated", flush=True)
