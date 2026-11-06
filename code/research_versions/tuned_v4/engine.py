"""Resumable source search, pretarget model fitting and closed final evaluation."""
from datetime import datetime,timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import traceback
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from nir_dg.datasets import ROOT
from nir_dg.series_v2.engine import verify_freeze
from nir_dg.series_v2.data import features,frame_hash
from research_versions.robustness_v3.preprocessing import make_preprocessor
from . import data
from .data import STORE,RUN,source,pack,matrix_hash
from .design import *
from .models import TunedModel,control_parameters

REPORT=ROOT/'reports/tuned_v4'


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8',newline='\n');tmp.replace(path)


def hashes():
    paths=[*sorted((ROOT/'research_versions').rglob('*.py')),*sorted((ROOT/'src/nir_dg').rglob('*.py')),ROOT/'tools/run_tuned_v4.py',ROOT/'pyproject.toml',ROOT/'uv.lock']
    return {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare():
    previous=verify_freeze();RUN.mkdir(exist_ok=True);STORE.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    path=RUN/'source_protocol.json'
    if path.exists():
        p=json.loads(path.read_text(encoding='utf-8'));assert p['protocol']['runtime_hashes']==hashes(),'Frozen v4 source runtime changed';return p
    # Assert prospective real domains were not downloaded in an older cache.
    forbidden={'psam_p26.csv','psam_p34.csv','psam_p55.csv','psam_p24.csv','psam_p09.csv'}
    already=[str(p) for p in (ROOT/'data').rglob('psam_p*.csv') if p.name in forbidden]
    if already:raise RuntimeError('Prospective ACS domain already present: '+str(already))
    checks=[]
    for c in contexts():
        f=source(c);folds=pack(f,c);joblib.dump(folds,STORE/f"packed_{c['id']}.joblib",compress=0)
        checks.append(dict(context=c,frame_hash=frame_hash(f),source_rows=len(f),source_domains=f.groupby('__env').size().to_dict(),packed_sha256=sha256((STORE/f"packed_{c['id']}.joblib").read_bytes()).hexdigest(),folds=[{k:v for k,v in fold.items() if k in ['transformed_train_sha256','transformed_validation_sha256']}|dict(train_rows=len(fold['y']),valid_rows=len(fold['valid_y'])) for fold in folds]))
    body=dict(series='tuned_v4',frozen_at=now(),previous_v2_protocol_sha256=previous['sha256'],runtime_hashes=hashes(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','cvxpy','clarabel','erasplit','anchorboosting','folktables']},contexts=checks,spaces={t:{m:candidates(m,t) for m in methods(t)} for t in ['classification','regression']},screening='24 candidates per method/context, first two fixed source folds, seed11, reduced tree/round budget; exact duplicate screening fits reused',promotion='all reference candidates plus four best nonreference candidates; four source folds, seeds11/47, full capacity',selection='maximum macro source fold/seed AUROC or negative MSE/full-development-source variance; smallest candidate ID breaks ties; any failed fold excludes candidate; expanded selection includes reference candidates',failure_policy='retain errors; no solver relaxation, no target-driven retry, no baseline substitution for failed methods',budget_qualification='equal candidate counts and promotion rule, not equal CPU time; per-fit durations retained; no global-optimum claim',new_acs_states=NEW_STATES,final_jobs=final_jobs(),final_protocol='all source selection and final source model files frozen and committed before new target access',statistics='24 primary tests in one Holm family: 15 tuned-minus-reference method/task gains and 9 tuned-minus-own-capacity-control gains; average scenarios equally then target domains/seeds, 10 independent MC data replicas; exact two-sided sign test with zeros omitted; marginal t95CI; real ACS descriptive only',hypothesis_scope='Conditional on one independently source-tuned configuration per development context; fresh final source draws, not per-replica retuning; new synthetic draws within known DGPs, not new mechanism families',real_scope='ACSIncome/PublicCoverage same survey family on five fresh states; Beijing/Appliances source HPO only because no fresh original real target remains',source_final_n=dict(synthetic=16384,acs=32000),workers=6,target_read=False)
    frozen=dict(protocol=body,sha256=digest(body));write(path,frozen);return frozen


def primary(task,y,p,scale):return float(roc_auc_score(y,p)) if task=='classification' else -float(np.mean(((y-p)/scale)**2))


def fit_cv(c,method,candidate,stage,packed,scale):
    path=RUN/'search'/c['id']/method/f"{stage}_{candidate['candidate']:02d}.json"
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    p=candidate['parameters'] if stage=='full' else screening_parameters(method,candidate['parameters'])
    result=dict(context=c['id'],method=method,candidate=candidate['candidate'],parameters=p,stage=stage,status='complete',folds=[]);tick=time.monotonic()
    try:
        for fi,fold in enumerate(packed if stage=='full' else packed[:2]):
            for model_seed in SEEDS if stage=='full' else [11]:
                model=TunedModel(method,c['task'],model_seed,p).fit(fold['X'],fold['y'],fold['env']);pred=model.predict(fold['V'])
                result['folds'].append(dict(fold=fi,seed=model_seed,primary=primary(c['task'],fold['valid_y'],pred,scale),prediction_sha256=sha256(pred.tobytes()).hexdigest()))
        result['mean_primary']=float(np.mean([r['primary'] for r in result['folds']]))
    except Exception:result['status']='failed';result['error']=traceback.format_exc()
    result['seconds']=time.monotonic()-tick;write(path,result);return result


def search_family(item):
    c,method=item;warnings.filterwarnings('ignore',category=DeprecationWarning)
    destination=RUN/'selection'/f"{c['id']}__{method}.json"
    if destination.exists():return json.loads(destination.read_text(encoding='utf-8'))
    frozen=json.loads((RUN/'source_protocol.json').read_text(encoding='utf-8'))
    check=next(s for s in frozen['protocol']['contexts'] if s['context']['id']==c['id'])
    file=STORE/f"packed_{c['id']}.joblib";assert sha256(file.read_bytes()).hexdigest()==check['packed_sha256']
    packed=joblib.load(file);f=source(c);scale=max(float(f.__y.std(ddof=0)),1e-8);grid=candidates(method,c['task']);screen=[];cache={}
    # No target loader used anywhere in this stage.
    data.synthetic_target=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('source stage target tripwire'))
    data.acs_target=data.synthetic_target
    for candidate in grid:
        key=digest(screening_parameters(method,candidate['parameters']))
        if key in cache:
            r={**cache[key],'candidate':candidate['candidate'],'screen_reused_from':cache[key]['candidate']}
            write(RUN/'search'/c['id']/method/f"screen_{candidate['candidate']:02d}.json",r)
        else:r=fit_cv(c,method,candidate,'screen',packed,scale);cache[key]=r
        screen.append(r)
    ranked=sorted([r for r in screen if r['status']=='complete' and not grid[r['candidate']]['reference']],key=lambda r:(-r['mean_primary'],r['candidate']))
    promoted=sorted({r['candidate'] for r in ranked[:4]}|{r['candidate'] for r in grid if r['reference']})
    full=[fit_cv(c,method,grid[i],'full',packed,scale) for i in promoted]
    good=sorted([r for r in full if r['status']=='complete'],key=lambda r:(-r['mean_primary'],r['candidate']))
    refs=[r for r in good if grid[r['candidate']]['reference']]
    if not good or not refs:raise RuntimeError('No valid method/reference candidate: '+c['id']+'/'+method)
    result=dict(context=c,method=method,selected=good[0]['candidate'],reference=refs[0]['candidate'],parameters=grid[good[0]['candidate']]['parameters'],reference_parameters=grid[refs[0]['candidate']]['parameters'],source_primary=good[0]['mean_primary'],reference_source_primary=refs[0]['mean_primary'],promoted=promoted,screen_failed=sum(r['status']!='complete' for r in screen),full_failed=sum(r['status']!='complete' for r in full),source_completed_at=now(),target_read=False)
    write(destination,result);return result


def source_fit(spec):
    warnings.filterwarnings('ignore',category=DeprecationWarning)
    receipt=RUN/'jobs'/spec['job_id']/'source.json'
    if receipt.exists():return json.loads(receipt.read_text(encoding='utf-8'))
    f=source(spec);y=f.__y.to_numpy();env=f.__env.to_numpy();p=make_preprocessor(f[features(f)]);X=p.fit_transform(f[features(f)]);models={};selected={};tick=time.monotonic()
    failures={}
    for method in methods(spec['task']):
        s=json.loads((RUN/'selection'/f"{spec['id']}__{method}.json").read_text(encoding='utf-8'));selected[method]=s
        try:
            pair={};tuned=TunedModel(method,spec['task'],spec['model_seed'],s['parameters']).fit(X,y,env);pair[method+':tuned']=tuned
            pair[method+':reference']=tuned if s['parameters']==s['reference_parameters'] else TunedModel(method,spec['task'],spec['model_seed'],s['reference_parameters']).fit(X,y,env)
            if method=='meta':pair[method+':control']=tuned.uniform_control()
            else:
                cp=control_parameters(method,s['parameters'])
                if cp is not None:pair[method+':control']=tuned if cp==s['parameters'] else TunedModel(method,spec['task'],spec['model_seed'],cp).fit(X,y,env)
            if method in {'irf','crf','maxrm'}:
                control=pair[method+':control']
                assert all(np.array_equal(a,b) for a,b in zip(tuned.model_.bootstrap_indices_,control.model_.bootstrap_indices_)),'Paired bootstrap mismatch'
                if method=='maxrm':
                    for a,b in zip(tuned.model_.members_,control.model_.members_):
                        assert np.array_equal(a.tree_.threshold,b.tree_.threshold),'Paired tree structure mismatch'
            models.update(pair)
        except Exception:failures[method]=traceback.format_exc()
    path=RUN/'models'/f"{spec['job_id']}.joblib";path.parent.mkdir(exist_ok=True);joblib.dump((p,models),path,compress=3)
    strong=max(BASELINES,key=lambda m:selected[m]['source_primary'])
    result=dict(spec=spec,source_frame_hash=frame_hash(f),source_rows=len(f),source_variance=float(np.var(y)),source_row_ids_sha256=sha256('\n'.join(f.__row_id).encode()).hexdigest(),transformed_source_sha256=matrix_hash(X),strong_baseline=strong,models_sha256=sha256(path.read_bytes()).hexdigest(),models=list(models),failed_methods=failures,selected_parameters={m:s['parameters'] for m,s in selected.items()},source_seconds=time.monotonic()-tick,source_completed_at=now(),target_read=False)
    write(receipt,result);return result


def seal():
    frozen=prepare();path=RUN/'final_protocol.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    selections={};receipts={}
    for c in contexts():
        for m in methods(c['task']):
            p=RUN/'selection'/f"{c['id']}__{m}.json";selections[p.relative_to(ROOT).as_posix()]=sha256(p.read_bytes()).hexdigest()
    for s in final_jobs():
        p=RUN/'jobs'/s['job_id']/'source.json';receipts[p.relative_to(ROOT).as_posix()]=sha256(p.read_bytes()).hexdigest()
    body=dict(series='tuned_v4',frozen_at=now(),source_protocol_sha256=frozen['sha256'],runtime_hashes=hashes(),selection_sha256=selections,source_receipt_sha256=receipts,jobs=final_jobs(),new_acs_states=NEW_STATES,statistics=frozen['protocol']['statistics'],scope=frozen['protocol']['hypothesis_scope'],target_policy='No previous real targets; new MC draws and five preregistered ACS states; no method changes/retries after opening',target_access_before_freeze=False)
    result=dict(protocol=body,sha256=digest(body));write(path,result);return result


def metrics(task,y,p):
    if not np.isfinite(p).all():raise RuntimeError('Nonfinite predictions')
    if task=='classification':
        auc=float(roc_auc_score(y,p));pos=np.asarray(y)==1;n1=pos.sum();n0=len(y)-n1
        independent=float((rankdata(p)[pos].sum()-n1*(n1+1)/2)/(n1*n0))
        if abs(auc-independent)>1e-12:raise RuntimeError('Independent AUROC mismatch')
        return dict(primary=auc,roc_auc=auc,average_precision=float(average_precision_score(y,p)),log_loss=float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),independent_auc_difference=abs(auc-independent))
    residual=np.asarray(y)-p;mse=float(residual@residual/len(y));independent=float(np.mean(np.square(y-p)))
    if abs(mse-independent)>1e-10*max(1,mse):raise RuntimeError('Independent MSE mismatch')
    return dict(primary=-mse,mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.mean(abs(residual))),independent_mse_difference=abs(mse-independent))


def evaluate(spec):
    path=RUN/'jobs'/spec['job_id'];done=path/'result.json'
    if done.exists():return json.loads(done.read_text(encoding='utf-8'))
    if (path/'target_access.json').exists():raise RuntimeError('Interrupted target job: fail closed; explicit audit needed')
    frozen=data.authorization();assert frozen['protocol']['runtime_hashes']==hashes(),'Final runtime changed'
    receipt_path=path/'source.json';relative=receipt_path.relative_to(ROOT).as_posix()
    assert sha256(receipt_path.read_bytes()).hexdigest()==frozen['protocol']['source_receipt_sha256'][relative]
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'));model_path=RUN/'models'/f"{spec['job_id']}.joblib"
    assert sha256(model_path.read_bytes()).hexdigest()==receipt['models_sha256'];processor,models=joblib.load(model_path)
    write(path/'target_access.json',dict(at=now(),final_protocol_sha256=frozen['sha256'],selection_frozen=True))
    target=data.synthetic_target(spec) if spec['kind']=='synthetic' else data.acs_target(spec['dataset'])
    src=source(spec);assert not set(src.__row_id)&set(target.__row_id)
    X=processor.transform(target[features(target)]);predictions={};rows=[]
    for name,model in models.items():
        pred=model.predict(X);predictions[name]=pred;method,variant=name.split(':')
        for domain in sorted(target.__env.unique()):
            mask=(target.__env==domain).to_numpy()
            rows.append(dict(job_id=spec['job_id'],context=spec['id'],task=spec['task'],kind=spec['kind'],scenario=spec['scenario'],data_replica=spec['data_replica'],model_seed=spec['model_seed'],method=method,variant=variant,target_domain=domain,source_variance=receipt['source_variance'],**metrics(spec['task'],target.__y.to_numpy()[mask],pred[mask])))
    values=target[['__row_id','__env','__y']].copy()
    for name,pred in predictions.items():values[name]=pred
    dest=RUN/'predictions'/f"{spec['job_id']}.parquet";dest.parent.mkdir(exist_ok=True);values.to_parquet(dest,index=False)
    result=dict(spec=spec,status='complete',target_rows=len(target),target_frame_hash=frame_hash(target),transformed_target_sha256=matrix_hash(X),transformed_feature_count=X.shape[1],predictions_sha256=sha256(dest.read_bytes()).hexdigest(),metrics=rows,strong_baseline=receipt['strong_baseline'],completed_at=now())
    write(done,result);return dict(job_id=spec['job_id'],status='complete',metric_rows=len(rows))
