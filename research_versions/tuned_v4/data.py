"""Development sources and sealed prospective targets in separate namespaces."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nir_dg.datasets import ROOT,ACS_SOURCE
from nir_dg.series_v2.data import source_real,features,frame_hash
from nir_dg.series_v2.generators import source_frame,draw,parameters
from research_versions.robustness_v3.preprocessing import canonical_frame,make_preprocessor
from .design import NEW_STATES,seed,digest

STORE=ROOT/'data/tuned_v4'
RUN=ROOT/'runs/tuned_v4'


def source(spec):
    return source_frame(spec) if spec['kind']=='synthetic' else source_real(spec)


def folds(frame,spec):
    domains=list(dict.fromkeys(frame.__env)) if spec['dataset'].startswith('acs_') else sorted(frame.__env.unique())
    if spec['dataset']=='appliances':
        chosen=['2016-W06','2016-W08','2016-W10','2016-W12']
        # Chronological expanding training with a 24h embargo.
        for domain in chosen:
            va=np.flatnonzero(frame.__env==domain)
            start=int(frame.iloc[va].__time.min())
            tr=np.flatnonzero((frame.__time<start-24*3600*10**9))
            if not len(tr) or not len(va):raise ValueError('Empty temporal fold')
            yield tr,va
    else:
        # First/last first: screening cannot secretly select easy folds.
        indices=[0,len(domains)-1,len(domains)//3,2*len(domains)//3]
        assert len(set(indices))==4
        for i in indices:
            yield np.flatnonzero(frame.__env!=domains[i]),np.flatnonzero(frame.__env==domains[i])


def matrix_hash(X):
    X=np.ascontiguousarray(X,dtype=np.float64)
    return sha256(str(X.shape).encode()+X.tobytes()).hexdigest()


def pack(frame,spec):
    result=[];y=frame.__y.to_numpy();env=frame.__env.to_numpy()
    for tr,va in folds(frame,spec):
        p=make_preprocessor(frame[features(frame)].iloc[tr]);X=p.fit_transform(frame[features(frame)].iloc[tr]);V=p.transform(frame[features(frame)].iloc[va])
        result.append(dict(X=X,V=V,y=y[tr],valid_y=y[va],env=env[tr],train_ids=frame.iloc[tr].__row_id.to_numpy(),valid_ids=frame.iloc[va].__row_id.to_numpy(),transformed_train_sha256=matrix_hash(X),transformed_validation_sha256=matrix_hash(V)))
    return result


def authorization():
    path=RUN/'final_protocol.json'
    if not path.exists():raise RuntimeError('Target is sealed until final protocol freeze')
    frozen=json.loads(path.read_text(encoding='utf-8'))
    if digest(frozen['protocol'])!=frozen['sha256']:raise RuntimeError('Final protocol hash mismatch')
    return frozen


def synthetic_target(spec):
    frozen=authorization()
    if spec not in frozen['protocol']['jobs']:raise RuntimeError('Unregistered target job')
    frames=[]
    for i,parameter in enumerate(parameters(spec['scenario'],spec['k'],'target')):
        rng=np.random.default_rng(seed('sealed_v4_target',spec['task'],spec['scenario'],spec['data_replica'],i))
        X,y=draw(spec['task'],spec['scenario'],2000,parameter,rng)
        f=pd.DataFrame(X,columns=[f'x{j}' for j in range(8)])
        f['__y']=y;f['__env']=f'v4_target_{i}'
        f['__row_id']=[f"v4:{spec['task']}:{spec['scenario']}:{spec['data_replica']}:{i}:{j}" for j in range(len(f))]
        frames.append(f)
    return canonical_frame(pd.concat(frames,ignore_index=True))


def acs_target(dataset):
    frozen=authorization()
    if dataset not in ['acs_income','acs_public_coverage'] or frozen['protocol']['new_acs_states']!=NEW_STATES:raise RuntimeError('Unregistered real target')
    from folktables import ACSDataSource,ACSIncome,ACSPublicCoverage
    task=ACSIncome if dataset=='acs_income' else ACSPublicCoverage
    cache=STORE/f'target_{dataset}.parquet'
    if cache.exists():return canonical_frame(pd.read_parquet(cache))
    frames=[];rawhashes={}
    for state in NEW_STATES:
        raw=ACSDataSource(survey_year='2018',horizon='1-Year',survey='person',root_dir=str(STORE/'sealed_acs')).get_data(states=[state],download=True)
        eligible=task._preprocess(raw);X,y,_=task.df_to_numpy(raw)
        # Independent eligibility and labels, separate from folktables transforms.
        if dataset=='acs_income':
            direct=raw[(raw.AGEP>16)&(raw.PINCP>100)&(raw.WKHP>0)&(raw.PWGTP>=1)]
            expected=(direct.PINCP>50000).to_numpy()
        else:
            direct=raw[(raw.AGEP<65)&(raw.PINCP<=30000)]
            expected=(direct.PUBCOV==1).to_numpy()
        direct_ids=direct.SERIALNO.astype(str)+':'+direct.SPORDER.astype(str)
        ids=eligible.SERIALNO.astype(str)+':'+eligible.SPORDER.astype(str)
        assert np.array_equal(expected,y) and np.array_equal(ids,direct_ids),'Independent raw label/ID mismatch'
        f=pd.DataFrame(X,columns=task.features)
        if 'ST' in f:f=f.drop(columns='ST')
        for c in f:
            if c not in {'AGEP','PINCP','WKHP','SCHL'}:f[c]=f[c].astype('Int64').astype(str)
        f['__y']=y.astype(int);f['__env']=state;f['__row_id']=(state+':'+ids).to_numpy();frames.append(f)
    ready=canonical_frame(pd.concat(frames,ignore_index=True));ready.to_parquet(cache,index=False)
    for p in (STORE/'sealed_acs').rglob('*.csv'):rawhashes[p.relative_to(ROOT).as_posix()]=sha256(p.read_bytes()).hexdigest()
    (RUN/f'raw_labels_{dataset}.json').write_text(json.dumps(dict(dataset=dataset,rows=len(ready),domains=ready.groupby('__env').size().to_dict(),independent_eligibility_labels_and_ids_match=True,raw_sha256=rawhashes),indent=2),encoding='utf-8',newline='\n')
    return canonical_frame(pd.read_parquet(cache))
