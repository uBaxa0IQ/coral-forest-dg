"""Distinguish stored-prediction mismatch from subset-batch preprocessing effects."""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[k]="1"
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from nir_dg.series_v2.data import features,STORE
from nir_dg.series_v2.engine import RUN,ROOT,write_json,now


if __name__=="__main__":
    specs=json.loads((RUN/"frozen_protocol.json").read_text(encoding="utf-8"))["protocol"]["jobs"]
    findings=[]
    for number in range(700,710):
        s=specs[number-1];job=s["job_id"]
        target=pd.read_parquet(STORE/f"target_{s['dataset']}.parquet")
        processor,models=joblib.load(RUN/"models"/f"{job}.joblib")
        saved=pd.read_parquet(RUN/"predictions"/f"{job}_target.parquet")
        ids=np.unique(np.linspace(0,len(target)-1,256).astype(int))
        small=processor.transform(target.iloc[ids][features(target)])
        full=None
        for key,model in models.items():
            method,ci=key.split(":");g=saved[(saved.method==method)&(saved.config==int(ci))]
            expected=g.prediction.to_numpy();p=model.predict(small)
            diff=float(np.max(np.abs(p-expected[ids])))
            if diff>1e-8:
                if full is None:
                    full=processor.transform(target[features(target)])
                actual=model.predict(full)
                row=dict(job_id=job,ordinal=number,dataset=s["dataset"],method=method,config=int(ci),subset_probe_difference=diff,full_batch_prediction_difference=float(np.max(np.abs(actual-expected))),full_transform_slice_prediction_difference=float(np.max(np.abs(model.predict(full[ids])-expected[ids]))),subset_full_input_difference=float(np.max(np.abs(small-full[ids]))))
                findings.append(row);print(json.dumps(row),flush=True)
        print("CHECKED",number,flush=True)
    write_json(ROOT/"reports/series_v2/model_replay_diagnostic.json",dict(at=now(),findings=findings,training_runtime_modified=False))
