from hashlib import sha256
import io
import json
import os
import time
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from nir_dg.datasets import ROOT, ACS_SOURCE, BEIJING_SOURCE, download
from .generators import allocation, seed_for

STORE=ROOT/"data/series_v2"
META=["__y","__env","__row_id","__time"]


def frame_hash(frame):
    h=sha256()
    h.update(json.dumps([(c,str(frame[c].dtype)) for c in frame],sort_keys=True).encode())
    h.update(pd.util.hash_pandas_object(frame,index=False).values.tobytes())
    return h.hexdigest()


def features(frame):
    return [c for c in frame if not c.startswith("__")]


def _authorize(role,authorization=None):
    if role not in {"source","target"}:
        raise ValueError("Role must be explicit")
    if role=="target":
        from .engine import verify_freeze
        if authorization is None or verify_freeze()["sha256"]!=authorization:
            raise RuntimeError("Target loader requires frozen authorization")


def acs_frame(state, problem_name, role,authorization=None):
    _authorize(role,authorization)
    from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage
    registry=json.loads((ROOT/"configs/target_registry.json").read_text())
    allowed=ACS_SOURCE if role=="source" else registry["reserved_v2_new_targets"]["acs_2018"]
    if state not in allowed:
        raise RuntimeError("Historical/unregistered ACS domain forbidden in v2")
    task=ACSIncome if problem_name=="acs_income" else ACSPublicCoverage
    raw=ACSDataSource(survey_year="2018",horizon="1-Year",survey="person",root_dir=str(ROOT/"data/raw/acs" if role=="source" else STORE/"sealed_acs")).get_data(states=[state],download=True)
    eligible=task._preprocess(raw)
    X,y,_=task.df_to_numpy(raw)
    f=pd.DataFrame(X,columns=task.features)
    if "ST" in f:
        f=f.drop(columns="ST")
    numeric={"AGEP","PINCP","WKHP","SCHL"}
    for c in f:
        if c not in numeric:
            f[c]=f[c].astype("Int64").astype(str)
    f["__y"],f["__env"]=y.astype(int),state
    assert len(eligible)==len(f)
    if "SERIALNO" in eligible and "SPORDER" in eligible:
        ids=eligible.SERIALNO.astype(str)+":"+eligible.SPORDER.astype(str)
        f["__row_id"]=(state+":"+ids).to_numpy()
    else:
        raise RuntimeError("Stable Census person identifier unavailable")
    return f


def beijing_frame(stations, role,authorization=None):
    _authorize(role,authorization)
    registry=json.loads((ROOT/"configs/target_registry.json").read_text())
    allowed=BEIJING_SOURCE if role=="source" else registry["reserved_v2_new_targets"]["beijing_station"]
    if not set(stations)<=set(allowed):
        raise RuntimeError("Historical/unregistered station forbidden in v2")
    frames=[]
    with zipfile.ZipFile(ROOT/"data/raw/beijing.zip") as outer:
        nested=[n for n in outer.namelist() if n.endswith(".zip")]
        inner=zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
        for station in stations:
            member=next(n for n in inner.namelist() if f"_{station}_" in n and n.endswith(".csv"))
            f=pd.read_csv(inner.open(member)).dropna(subset=["PM2.5"])
            f["__y"],f["__env"],f["__row_id"]=f["PM2.5"],station,station+":"+f.No.astype(str)
            f=f[["year","month","day","hour","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","wd","WSPM","__y","__env","__row_id"]]
            frames.append(f)
        if inner is not outer:
            inner.close()
    return pd.concat(frames,ignore_index=True)


def prepare_appliances():
    source=STORE/"appliances_source.csv"
    if source.exists():
        return
    archive=download("https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",STORE/"raw/appliances.zip")
    target=STORE/"sealed_appliances.csv"
    source.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z, source.open("wb") as s, target.open("wb") as t:
        member=next(n for n in z.namelist() if n.endswith("energydata_complete.csv"))
        with z.open(member) as raw:
            header=raw.readline(); s.write(header); t.write(header)
            for opaque in raw:
                date=opaque.split(b",",1)[0].strip(b'"')[:10]
                if date<b"2016-03-31":
                    s.write(opaque)
                elif b"2016-04-02"<=date<=b"2016-05-27":
                    t.write(opaque)


def appliances_frame(role,authorization=None):
    _authorize(role,authorization)
    if role=="source":
        prepare_appliances()
    path=STORE/("appliances_source.csv" if role=="source" else "sealed_appliances.csv")
    f=pd.read_csv(path)
    time=pd.to_datetime(f.date)
    X=f[[*(f"T{i}" for i in range(1,10)),*(f"RH_{i}" for i in range(1,10))]].copy()
    hour=time.dt.hour+time.dt.minute/60
    X["hour_sin"],X["hour_cos"]=np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)
    X["dow_sin"],X["dow_cos"]=np.sin(2*np.pi*time.dt.dayofweek/7),np.cos(2*np.pi*time.dt.dayofweek/7)
    X["__y"],X["__row_id"],X["__time"]=f.Appliances,time.astype(str),time.astype("int64")
    X["__env"]=(time.dt.strftime("%G-W%V") if role=="source" else time.dt.strftime("%Y-%m"))
    if role=="source":
        counts=X.groupby("__env").size()
        X=X[X.__env.isin(counts[counts>=950].index)]
    return X.reset_index(drop=True)


def prepare_source_pools():
    STORE.mkdir(parents=True,exist_ok=True)
    summaries={}
    for name in ["acs_income","acs_public_coverage","beijing_pm25","appliances"]:
        path=STORE/f"pool_{name}.parquet"
        if not path.exists():
            if name.startswith("acs_"):
                f=pd.concat([acs_frame(s,name,"source") for s in ACS_SOURCE],ignore_index=True)
            elif name=="beijing_pm25":
                f=beijing_frame(BEIJING_SOURCE,"source")
            else:
                f=appliances_frame("source")
            f.to_parquet(path,index=False)
        f=pd.read_parquet(path)
        summaries[name]=dict(rows=len(f),domains=f.groupby("__env").size().to_dict(),features=features(f),frame_hash=frame_hash(f),file_sha256=sha256(path.read_bytes()).hexdigest(),target_read=False)
        print(name,summaries[name]["rows"],flush=True)
    (STORE/"source_pool_manifest.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    return summaries


def source_real(spec):
    f=pd.read_parquet(STORE/f"pool_{spec['dataset']}.parquet")
    domains=list(dict.fromkeys(f.__env)) if spec["dataset"].startswith("acs_") else sorted(f.__env.unique())
    take=np.unique(np.linspace(0,len(domains)-1,spec["k"]).astype(int))
    domains=[domains[i] for i in take]
    if len(domains)!=spec["k"]:
        raise ValueError("Requested more source domains than available")
    counts=allocation(spec["n"],spec["k"],spec["imbalance"],spec["replica"])
    parts=[]
    for domain,n in zip(domains,counts):
        g=f[f.__env==domain]
        if len(g)<n:
            raise ValueError(f"Source-only incompatibility: {domain} has {len(g)}, needs {n}")
        order=np.random.default_rng(seed_for("real_source",spec["dataset"],spec["replica"],domain)).permutation(len(g))
        parts.append(g.iloc[order[:n]])
    return pd.concat(parts,ignore_index=True)


def target_real(spec,authorization=None):
    from .engine import verify_freeze
    if authorization is None or verify_freeze()["sha256"] != authorization:
        raise RuntimeError("New real target requires frozen protocol authorization")
    cache=STORE/f"target_{spec['dataset']}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    lock=STORE/f"target_{spec['dataset']}.lock"
    start=time.monotonic()
    while True:
        try:
            descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic()-start>1800:
                raise RuntimeError("Target loading lock timed out; inspect interrupted loader")
            time.sleep(1)
    try:
        if cache.exists():
            return pd.read_parquet(cache)
        f=_target_real_uncached(spec,authorization)
        f.to_parquet(cache,index=False)
        return f
    finally:
        lock.unlink()


def _target_real_uncached(spec,authorization):
    registry=json.loads((ROOT/"configs/target_registry.json").read_text())
    if spec["dataset"].startswith("acs_"):
        return pd.concat([acs_frame(s,spec["dataset"],"target",authorization) for s in registry["reserved_v2_new_targets"]["acs_2018"]],ignore_index=True)
    if spec["dataset"]=="beijing_pm25":
        return beijing_frame(registry["reserved_v2_new_targets"]["beijing_station"],"target",authorization)
    return appliances_frame("target",authorization)
