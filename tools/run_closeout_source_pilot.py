"""Bounded source-only feasibility pilot; no target construction or evaluation."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from nir_dg.series_v2.generators import source_frame
from nir_dg.series_v2.data import features,frame_hash
from nir_dg.series_v2.engine import folds
from research_versions.robustness_v3.preprocessing import make_preprocessor
from research_versions.robustness_v3.tuning import ConfiguredModel,configurations

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/closeout/source_pilot'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    tmp.replace(path)


def forbidden_target(*args,**kwargs):
    raise RuntimeError('Target is forbidden in the source-only pilot')


def run_one(item):
    # Logical tripwire, not an OS sandbox or an attestation of all possible I/O.
    import nir_dg.series_v2.generators as g
    import nir_dg.series_v2.data as d
    import nir_dg.series_v2.engine as e
    g.target_frame=d.target_real=e.target_frame=e.target_real=forbidden_target
    spec,cid,config=item
    path=OUT/'jobs'/f"{spec['task']}_{cid:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    started=time.monotonic();result=dict(task=spec['task'],config_id=cid,config=config,status='complete',folds=[])
    try:
        f=source_frame(spec);y=f.__y.to_numpy();env=f.__env.to_numpy();scale=max(float(y.std()),1e-8)
        for fi,(tr,va) in enumerate(folds(f,spec)):
            p=make_preprocessor(f[features(f)].iloc[tr]);a=p.fit_transform(f[features(f)].iloc[tr]);b=p.transform(f[features(f)].iloc[va])
            model=ConfiguredModel(config['method'],spec['task'],11,dict(profile=config['profile'],**config['parameters'])).fit(a,y[tr],env[tr])
            prediction=model.predict(b)
            value=float(roc_auc_score(y[va],prediction)) if spec['task']=='classification' else -float(np.mean(((y[va]-prediction)/scale)**2))
            fold=dict(fold=fi,primary=value,train_rows=len(tr),validation_rows=len(va),prediction_sha256=sha256(prediction.tobytes()).hexdigest())
            if hasattr(model.model_,'certificates_') and model.model_.certificates_:
                fold['max_primal_dual_gap']=max(v['gap'] for v in model.model_.certificates_)
            if hasattr(model.model_,'members_'):
                fold['member_count']=len(model.model_.members_)
            result['folds'].append(fold)
        result['mean_source_primary']=float(np.mean([v['primary'] for v in result['folds']]))
    except Exception:
        result['status']='failed';result['error']=traceback.format_exc()
    result['seconds']=time.monotonic()-started;write(path,result)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=[];contexts=[]
    for task in ['classification','regression']:
        spec=dict(kind='synthetic',dataset='synthetic',task=task,scenario='spurious',k=8,n=4096,imbalance=1.,replica=1001,model_seed=11)
        f=source_frame(spec)
        contexts.append(dict(spec=spec,source_hash=frame_hash(f),folds=[dict(train_rows=len(tr),validation_rows=len(va)) for tr,va in folds(f,spec)]))
        tasks.extend((spec,i,c) for i,c in enumerate(configurations(task)))
    files=[Path(__file__),*sorted((ROOT/'research_versions').rglob('*.py'))]
    body=dict(purpose='exploratory source-only capacity/solver feasibility; not best OOD comparison',contexts=contexts,configurations={t:configurations(t) for t in ['classification','regression']},selection='largest mean source-heldout AUROC / negative MSE divided by full-source variance; order breaks ties; any failed fold excludes config',workers=2,model_seeds=[11],target_evaluations=0,target_tripwire='generator and real target entrypoints replaced with raising functions inside workers; not an OS security guarantee',failure_policy='retain failures, no quality-based retry/fallback',maxrm_certificate_tolerance='1e-6 * max(1, primal); solver status optimal required',code_hashes={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in files})
    manifest=OUT/'source_protocol.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        assert old['protocol']==body,'Source pilot protocol/code changed'
    else:
        write(manifest,dict(frozen_at=datetime.now(timezone.utc).isoformat(),protocol=body,sha256=sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
    started=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            print(json.dumps(dict(task=r['task'],config=r['config_id'],status=r['status'],seconds=round(r['seconds'],2))),flush=True)
    rows=[]
    for r in results:
        rows.append(dict(task=r['task'],config_id=r['config_id'],method=r['config']['method'],profile=r['config']['profile'],parameters=json.dumps(r['config']['parameters'],sort_keys=True),status=r['status'],mean_source_primary=r.get('mean_source_primary'),seconds=r['seconds']))
    table=pd.DataFrame(rows).sort_values(['task','config_id']);table.to_csv(OUT/'configuration_results.csv',index=False)
    complete=table[table.status=='complete'];selected=complete.sort_values(['task','method','mean_source_primary','config_id'],ascending=[True,True,False,True]).groupby(['task','method'],sort=False).head(1)
    selected.to_csv(OUT/'selected_source_configurations.csv',index=False)
    write(OUT/'summary.json',dict(configurations=len(table),complete_configurations=len(complete),failed_configurations=int((table.status!='complete').sum()),completed_source_folds=sum(len(r['folds']) for r in results),wall_seconds=time.monotonic()-started,fit_seconds=float(table.seconds.sum()),target_evaluations=0,conclusion='Source feasibility only; no claim about target benefit, optimal hyperparameters or equal compute budgets'))


if __name__=='__main__':
    main()
