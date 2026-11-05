"""Post-evaluation forensic sensitivity; never refits or replaces official scores."""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[k]="1"
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from nir_dg.metrics import score
from nir_dg.series_v2.data import STORE,features
from nir_dg.series_v2.engine import ROOT,RUN,write_json,now


if __name__=="__main__":
    job="5259a196ef641ff1"
    f=pd.read_parquet(STORE/"target_beijing_pm25.parquet")
    processor,models=joblib.load(RUN/"models"/f"{job}.joblib")
    recorded=pd.read_parquet(RUN/"predictions"/f"{job}_target.parquet")
    cached=processor.transform(f[features(f)])
    raw=f.copy();raw.loc[raw.wd.isna(),"wd"]=np.nan
    original=processor.transform(raw[features(raw)])
    y=f.__y.to_numpy();env=f.__env.to_numpy();rows=[];replays=[]
    for key,model in models.items():
        method,config=key.split(":");config=int(config)
        actual=model.predict(original);alternative=model.predict(cached)
        saved=recorded[(recorded.method==method)&(recorded.config==config)].prediction.to_numpy()
        difference=float(np.max(np.abs(actual-saved)));assert difference<=1e-8
        replays.append(dict(method=method,config=config,full_vector_original_input_max_abs_difference=difference,cached_input_changed_rows=int(np.count_nonzero(np.abs(alternative-saved)>1e-8))))
        for e in np.unique(env):
            mask=env==e;a=score("regression",y[mask],actual[mask]);b=score("regression",y[mask],alternative[mask])
            rows.append(dict(method=method,config=config,target_domain=e,original_rmse=a["rmse"],cache_reload_rmse=b["rmse"],cache_minus_original_rmse=b["rmse"]-a["rmse"],original_mse=a["mse"],cache_reload_mse=b["mse"]))
    out=ROOT/"reports/series_v2/audit"
    pd.DataFrame(rows).to_csv(out/"null_cache_sensitivity.csv",index=False)
    write_json(out/"null_cache_full_vector_replay.json",dict(at=now(),job_id=job,target_rows=len(f),missing_wd_rows=int(f.wd.isna().sum()),models=replays,official_predictions_and_metrics_unchanged=True,post_evaluation_forensic_diagnostic=True,max_abs_domain_rmse_change=float(max(abs(r["cache_minus_original_rmse"]) for r in rows))))
    print(json.dumps(replays,indent=2),flush=True)
