"""Source-only data loaders. No target loader is exposed by this module."""
from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests
from nir_dg.protocol import ProtocolError, SourceData

ROOT = Path(__file__).resolve().parents[2]
ACS_SOURCE = ["CA", "TX", "NY", "FL", "IL", "WA", "GA", "AZ"]
ACS_TARGET = ["CO", "NC", "MA"]
BEIJING_SOURCE = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng"]
BEIJING_TARGET = ["Huairou", "Shunyi"]


def download(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=(20, 120))
        response.raise_for_status()
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_bytes(response.content)
        partial.replace(path)
        path.with_suffix(path.suffix + ".provenance.json").write_text(json.dumps(dict(url=url, sha256=sha256(response.content).hexdigest()), indent=2), encoding="utf-8")
    return path


def sample_domains(frame, domain_col, cap, seed=20261104):
    if cap is None:
        return frame.reset_index(drop=True)
    return pd.concat([g.sample(min(len(g), cap), random_state=seed) for _, g in frame.groupby(domain_col, sort=True)], ignore_index=True)


def save_source(data):
    path = ROOT / "data/source" / data.name
    path.mkdir(parents=True, exist_ok=True)
    data.X.to_parquet(path / "X.parquet", index=False)
    arrays = dict(y=data.y, environments=data.environments.astype(str), row_ids=data.row_ids.astype(str))
    if data.time_ids is not None:
        arrays["time_ids"] = data.time_ids
    if data.entity_ids is not None:
        arrays["entity_ids"] = data.entity_ids.astype(str)
    np.savez_compressed(path / "arrays.npz", **arrays)
    meta = dict(name=data.name, task=data.task, temporal=data.temporal, source_rows=len(data.y), feature_names=list(data.X), source_domains={str(e): int(np.sum(data.environments == e)) for e in np.unique(data.environments)}, source_fingerprint=data.fingerprint())
    if data.task == "classification":
        meta["source_prevalence"] = {str(e): float(data.y[data.environments == e].mean()) for e in np.unique(data.environments)}
    (path / "source_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_source(name):
    if name not in {"acs_income", "hospital_readmission", "beijing_pm25", "gas_turbine_co", "synthetic_classification", "synthetic_regression"}:
        raise ProtocolError("Dataset is not registered as source")
    path = ROOT / "data/source" / name
    meta = json.loads((path / "source_metadata.json").read_text(encoding="utf-8"))
    with np.load(path / "arrays.npz", allow_pickle=False) as values:
        data = SourceData(pd.read_parquet(path / "X.parquet"), values["y"], values["environments"], values["row_ids"], meta["task"], name, meta["temporal"], values["time_ids"] if "time_ids" in values else None, values["entity_ids"] if "entity_ids" in values else None)
    if data.fingerprint() != meta["source_fingerprint"]:
        raise ProtocolError("Stored source data fingerprint mismatch")
    return data


def prepare_acs(cap=400):
    from folktables import ACSDataSource, ACSIncome
    frames = []
    source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=str(ROOT / "data/raw/acs"))
    for state in ACS_SOURCE:
        # Target state downloads cannot be requested through this source function.
        frame = source.get_data(states=[state], download=True)
        X, y, _ = ACSIncome.df_to_numpy(frame)
        ready = pd.DataFrame(X, columns=ACSIncome.features)
        ready["y"] = y.astype(int)
        ready["env"] = state
        ready["row_id"] = [f"{state}:{i}" for i in range(len(ready))]
        frames.append(sample_domains(ready, "env", cap))
        print(f"ACS source {state}: {len(ready)} eligible, sampled {len(frames[-1])}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    features = ACSIncome.features
    X = frame[features].copy()
    # PUMS category codes are nominal, using the official task's feature list.
    for col in ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]:
        X[col] = X[col].astype("Int64").astype(str)
    return save_source(SourceData(X, frame.y.to_numpy(), frame.env.to_numpy(), frame.row_id.to_numpy(), "classification", "acs_income"))


def prepare_hospital(cap=400):
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    archive = download(url, ROOT / "data/raw/hospital.zip")
    source_path = ROOT / "data/raw/hospital_source.csv"
    target_path = ROOT / "data/target_locked/hospital.csv"
    if not source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if n.endswith("diabetic_data.csv")]
            if len(names) != 1:
                raise ProtocolError("Unexpected UCI archive layout")
            with z.open(names[0]) as raw, source_path.open("wb") as source, target_path.open("wb") as target:
                header = raw.readline()
                domain_index = header.decode("utf-8-sig").strip().split(",").index("admission_source_id")
                source.write(header)
                target.write(header)
                for opaque_row in raw:
                    # Stateless mechanical routing reads the published domain field only.
                    # No target feature/label objects, counts, statistics or transformations.
                    domain = opaque_row.split(b",", domain_index + 1)[domain_index]
                    (target if domain == b"7" else source).write(opaque_row)
    frame = pd.read_csv(source_path, na_values=["?"])
    frame = frame.dropna(subset=["race"])
    # Keep the first row per source patient to avoid pseudoreplication. Encounter order
    # is dataset row order, not claimed to be a verified chronological timestamp.
    frame = frame.drop_duplicates("patient_nbr", keep="first")
    # Rule based on source counts/classes only. No merging of unrelated tiny groups.
    allowed = [e for e, g in frame.groupby("admission_source_id") if e not in {9, 15, 17, 20, 21} and len(g) >= 150 and (g.readmitted != "NO").nunique() == 2]
    frame = frame[frame.admission_source_id.isin(allowed)].copy()
    frame["env"] = frame.admission_source_id.astype(str)
    frame = sample_domains(frame, "env", cap)
    # A compact, explicit subset available by discharge. Not an exact TableShift
    # feature-pipeline replication. ID, domain and diagnoses/high-cardinality text excluded.
    numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    categorical = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed"]
    X = frame[numeric + categorical].copy()
    for column in categorical:
        X[column] = X[column].fillna("MISSING").astype(str)
    return save_source(SourceData(X, (frame.readmitted != "NO").to_numpy(dtype=int), frame.env.to_numpy(), frame.encounter_id.to_numpy().astype(str), "classification", "hospital_readmission", entity_ids=frame.patient_nbr.to_numpy().astype(str)))


def prepare_beijing(cap=400):
    url = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
    archive = download(url, ROOT / "data/raw/beijing.zip")
    frames = []
    with zipfile.ZipFile(archive) as outer:
        nested = [n for n in outer.namelist() if n.endswith(".zip")]
        context = zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
        try:
            for station in BEIJING_SOURCE:
                name = next(n for n in context.namelist() if n.endswith(".csv") and f"_{station}_" in n)
                frame = pd.read_csv(context.open(name)).dropna(subset=["PM2.5"])
                frame["env"] = station
                frame["row_id"] = station + ":" + frame.No.astype(str)
                frames.append(sample_domains(frame, "env", cap))
        finally:
            if context is not outer:
                context.close()
    frame = pd.concat(frames, ignore_index=True)
    features = ["year", "month", "day", "hour", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"]
    return save_source(SourceData(frame[features].copy(), frame["PM2.5"].to_numpy(), frame.env.to_numpy(), frame.row_id.to_numpy(), "regression", "beijing_pm25"))


def prepare_synthetic(task, cap=400, seed=20261104):
    """Mechanism test; never substituted for an independent real-world task."""
    rng = np.random.default_rng(seed)
    frames = []
    for e, spurious_scale in enumerate([2., 1., .3, -1.]):
        stable = rng.normal(size=cap)
        noise = rng.normal(size=cap)
        y = (stable + .5 * noise > 0).astype(int) if task == "classification" else 2 * stable + .5 * noise
        spurious = spurious_scale * y + rng.normal(size=cap)
        frames.append(pd.DataFrame(dict(stable=stable, spurious=spurious, noise=rng.normal(size=cap), y=y, env=str(e), row_id=[f"{e}:{i}" for i in range(cap)])))
    frame = pd.concat(frames, ignore_index=True)
    return save_source(SourceData(frame[["stable", "spurious", "noise"]], frame.y.to_numpy(), frame.env.to_numpy(), frame.row_id.to_numpy(), task, "synthetic_" + task))


def prepare_gas(cap=400):
    # Follow the official first-three-years training / last-two-years test rule.
    url = "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip"
    archive = download(url, ROOT / "data/raw/gas_turbine.zip")
    frames = []
    with zipfile.ZipFile(archive) as z:
        for year in [2011, 2012, 2013]:
            name = next(n for n in z.namelist() if n.endswith(f"gt_{year}.csv"))
            frame = pd.read_csv(z.open(name))
            frame["env"] = str(year)
            frame["time_id"] = year
            frame["row_id"] = [f"{year}:{i}" for i in range(len(frame))]
            frames.append(sample_domains(frame, "env", cap))
    frame = pd.concat(frames, ignore_index=True)
    features = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
    return save_source(SourceData(frame[features], frame.CO.to_numpy(), frame.env.to_numpy(), frame.row_id.to_numpy(), "regression", "gas_turbine_co", True, frame.time_id.to_numpy()))
