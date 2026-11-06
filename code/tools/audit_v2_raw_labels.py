"""Independent raw-file reconstruction of eligible target IDs and outcomes.

Does not call the experimental data loaders or folktables task transformations.
Explicit allowlists restrict reads to the already registered v2 target domains.
"""
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT/"data/series_v2"


def compare(name,raw):
    cached=pd.read_parquet(STORE/f"target_{name}.parquet",columns=["__row_id","__y","__env"])
    assert raw.__row_id.is_unique and cached.__row_id.is_unique
    assert set(raw.__row_id)==set(cached.__row_id)
    expected=raw.set_index("__row_id").loc[cached.__row_id]
    assert np.array_equal(expected.__y.to_numpy(),cached.__y.to_numpy())
    assert np.array_equal(expected.__env.to_numpy(),cached.__env.to_numpy())
    return dict(dataset=name,rows=len(raw),domains=raw.groupby("__env").size().to_dict(),row_ids_and_y_match=True)


if __name__=="__main__":
    checks=[];hashes={};acs={"acs_income":[],"acs_public_coverage":[]}
    for state,code in [("MN","27"),("OH","39"),("PA","42"),("OR","41"),("VA","51")]:
        p=STORE/"sealed_acs/2018/1-Year"/f"psam_p{code}.csv"
        hashes[p.relative_to(ROOT).as_posix()]=sha256(p.read_bytes()).hexdigest()
        f=pd.read_csv(p,usecols=["AGEP","PINCP","WKHP","PWGTP","PUBCOV","SERIALNO","SPORDER"])
        for name in acs:
            valid=((f.AGEP>16)&(f.PINCP>100)&(f.WKHP>0)&(f.PWGTP>=1)) if name=="acs_income" else ((f.AGEP<65)&(f.PINCP<=30000))
            g=f[valid]
            y=(g.PINCP>50000) if name=="acs_income" else (g.PUBCOV==1)
            acs[name].append(pd.DataFrame({"__row_id":state+":"+g.SERIALNO.astype(str)+":"+g.SPORDER.astype(str),"__y":y.astype(int),"__env":state}))
    for name,parts in acs.items():
        checks.append(compare(name,pd.concat(parts,ignore_index=True)))
    p=ROOT/"data/raw/beijing.zip";hashes[p.relative_to(ROOT).as_posix()]=sha256(p.read_bytes()).hexdigest()
    parts=[]
    with zipfile.ZipFile(p) as outer:
        nested=[n for n in outer.namelist() if n.endswith(".zip")]
        inner=zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
        for station in ["Nongzhanguan","Tiantan","Wanliu","Wanshouxigong"]:
            members=[n for n in inner.namelist() if f"_{station}_" in n and n.endswith(".csv")];assert len(members)==1
            f=pd.read_csv(inner.open(members[0]),usecols=["No","PM2.5"]).dropna(subset=["PM2.5"])
            parts.append(pd.DataFrame({"__row_id":station+":"+f.No.astype(str),"__y":f["PM2.5"],"__env":station}))
        if inner is not outer:
            inner.close()
    checks.append(compare("beijing_pm25",pd.concat(parts,ignore_index=True)))
    p=STORE/"raw/appliances.zip";hashes[p.relative_to(ROOT).as_posix()]=sha256(p.read_bytes()).hexdigest()
    with zipfile.ZipFile(p) as archive:
        member=next(n for n in archive.namelist() if n.endswith("energydata_complete.csv"))
        with archive.open(member) as file:
            header=file.readline();selected=[]
            for opaque in file:
                date=opaque.split(b",",1)[0].strip(b'"')[:10]
                if b"2016-04-02"<=date<=b"2016-05-27":
                    selected.append(opaque)
    f=pd.read_csv(io.BytesIO(header+b"".join(selected)),usecols=["date","Appliances"])
    time=pd.to_datetime(f.date)
    raw=pd.DataFrame({"__row_id":time.dt.strftime("%Y-%m-%d %H:%M:%S"),"__y":f.Appliances,"__env":time.dt.strftime("%Y-%m")})
    checks.append(compare("appliances",raw))
    result=dict(status="passed",independence="Separate raw CSV/ZIP reconstruction, no experimental loader or task transformation called",checks=checks,raw_file_sha256=hashes,script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),unregistered_gap_outcomes_not_parsed=True)
    out=ROOT/"reports/series_v2/audit/raw_target_labels.json";out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
