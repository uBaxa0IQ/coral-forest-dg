"""Small saved-model smoke check after relocation; never trains or writes history."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'code'
sys.path[:0]=[str(CODE),str(CODE/'src')]
from research_versions.tuned_v4.data import synthetic_target
from nir_dg.series_v2.data import features


def main():
    frozen=json.loads((CODE/'runs/tuned_v4/final_protocol.json').read_text(encoding='utf-8'))
    checks=[]
    for task in ['classification','regression']:
        s=next(s for s in frozen['protocol']['jobs'] if s['kind']=='synthetic' and s['task']==task)
        receipt=json.loads((CODE/f"runs/tuned_v4/jobs/{s['job_id']}/source.json").read_text(encoding='utf-8'))
        path=CODE/f"runs/tuned_v4/models/{s['job_id']}.joblib"
        assert sha256(path.read_bytes()).hexdigest()==receipt['models_sha256']
        processor,models=joblib.load(path)
        # Deterministic replay of a previously opened, completed job. No new seed.
        frame=synthetic_target(s);X=processor.transform(frame[features(frame)].iloc[:64])
        saved=pd.read_parquet(CODE/f"runs/tuned_v4/predictions/{s['job_id']}.parquet").iloc[:64]
        delta=max(float(np.max(abs(m.predict(X)-saved[name].to_numpy()))) for name,m in models.items())
        assert delta==0.,(task,delta)
        checks.append(dict(task=task,job_id=s['job_id'],models=len(models),rows=64,max_prediction_difference=delta))
    value=dict(scope='two saved jobs after path/environment relocation; not a new generalization experiment',checks=checks,old_artifacts_modified=False,new_target_domains_or_seeds_opened=False)
    (ROOT/'materials/repository_layout/model_probe.json').write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(value))


if __name__=='__main__':main()
