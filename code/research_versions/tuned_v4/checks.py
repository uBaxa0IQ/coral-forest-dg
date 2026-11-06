"""Predetermined reproducibility checks, with no hyperparameter decisions."""
from hashlib import sha256
import json
import joblib
import numpy as np
import pandas as pd
from nir_dg.series_v2.data import features
from . import data
from .data import STORE,RUN
from .models import TunedModel
from .engine import ROOT,REPORT,write


def replay_source(item):
    c,method=item
    path=RUN/'source_replays'/f"{c['id']}__{method}.json"
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    selected=json.loads((RUN/'selection'/f"{c['id']}__{method}.json").read_text(encoding='utf-8'))
    fold=joblib.load(STORE/f"packed_{c['id']}.joblib")[0]
    m=TunedModel(method,c['task'],11,selected['parameters']).fit(fold['X'],fold['y'],fold['env'])
    value=sha256(m.predict(fold['V']).tobytes()).hexdigest()
    original=json.loads((RUN/'search'/c['id']/method/f"full_{selected['selected']:02d}.json").read_text(encoding='utf-8'))
    assert value==original['folds'][0]['prediction_sha256'],(c['id'],method)
    result=dict(context=c['id'],method=method,candidate=selected['selected'],source_fold=0,seed=11,source_predictions_bytes_identical=True)
    write(path,result);return result


def replay_model(spec):
    p=RUN/'jobs'/spec['job_id']/'result.json';r=json.loads(p.read_text(encoding='utf-8'))
    target=data.synthetic_target(spec) if spec['kind']=='synthetic' else data.acs_target(spec['dataset'])
    processor,models=joblib.load(RUN/'models'/f"{spec['job_id']}.joblib")
    probe=processor.transform(target[features(target)].iloc[:256])
    saved=pd.read_parquet(RUN/'predictions'/f"{spec['job_id']}.parquet").iloc[:256]
    maximum=0.
    for name,m in models.items():
        difference=float(np.max(abs(m.predict(probe)-saved[name].to_numpy())));maximum=max(maximum,difference)
        assert difference<=1e-12,(spec['job_id'],name,difference)
    result=dict(job_id=spec['job_id'],models=len(models),probe_rows=len(saved),max_prediction_difference=maximum)
    write(RUN/'model_replays'/f"{spec['job_id']}.json",result);return result
