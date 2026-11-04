"""All contexts are enumerated before any new target value is observed."""
import json
from hashlib import sha256
from .generators import SCENARIOS


def identifier(spec):
    return sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:16]


def contexts():
    conditions=[]
    for task in ["classification","regression"]:
        for scenario in SCENARIOS:
            if task=="regression" and scenario=="label_prior":
                continue
            cells=[(8,8192,1)]
            if scenario in {"spurious","heteroscedastic"}:
                cells += [(4,8192,1),(16,8192,1),(8,2048,1),(8,8192,8)]
            elif scenario=="iid":
                cells += [(4,8192,1),(16,8192,1)]
            for k,n,imb in cells:
                for replica in range(10):
                    conditions.append(dict(kind="synthetic",dataset="synthetic",task=task,scenario=scenario,k=k,n=n,imbalance=imb,replica=replica))
    for dataset,task,ks,ns in [("acs_income","classification",[4,8],[4000,16000]),("acs_public_coverage","classification",[4,8],[4000,16000]),("beijing_pm25","regression",[4,6],[2400,9600]),("appliances","regression",[4,8],[2000,4000])]:
        cells=[(k,n) for k in ks for n in ns]
        if dataset=="appliances":
            cells.append((8,8000))
        for k,n in cells:
            for replica in range(3):
                conditions.append(dict(kind="real",dataset=dataset,task=task,scenario="observed",k=k,n=n,imbalance=1,replica=replica))
    return [{**s,"context_id":identifier(s)} for s in conditions]


def jobs():
    return [{**s,"model_seed":seed,"job_id":identifier({**s,"model_seed":seed})} for s in contexts() for seed in [11,47]]
