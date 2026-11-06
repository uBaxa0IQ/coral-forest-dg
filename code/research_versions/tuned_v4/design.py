"""Fixed search budgets and prospective data domains; no target I/O."""
from hashlib import sha256
import json
import numpy as np

ROOT_SEED=2026110504
NEW_STATES=['MI','NJ','WI','MD','CT']
SEEDS=[11,47]
REPLICAS=10
BASELINES=['rf','extra_trees','hist_gbdt']


def digest(x):return sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seed(*parts):return int(digest([ROOT_SEED,*parts])[:8],16)%(2**32-1)


def methods(task):return BASELINES+['irf','anchor']+(['crf','meta'] if task=='classification' else ['maxrm','era','directional_era'])


def contexts():
    result=[]
    for task in ['classification','regression']:
        for scenario in ['iid','covariate','spurious','concept','heteroscedastic','missingness']+(['label_prior'] if task=='classification' else []):
            result.append(dict(id=f'{task}_{scenario}',kind='synthetic',dataset='synthetic',task=task,scenario=scenario,k=8,n=8192,imbalance=1.,replica=4001))
    for dataset,task,k,n in [('acs_income','classification',8,16000),('acs_public_coverage','classification',8,16000),('beijing_pm25','regression',6,9600),('appliances','regression',8,8000)]:
        result.append(dict(id=dataset,kind='real',dataset=dataset,task=task,scenario='observed',k=k,n=n,imbalance=1.,replica=4001))
    return result


def references(method,task):
    if method in {'rf','extra_trees'}:return [dict(n_estimators=200,max_depth=None,min_samples_leaf=5,max_features='sqrt' if task=='classification' else 1.)]
    if method=='hist_gbdt':return [dict(max_iter=200,max_leaf_nodes=15,min_samples_leaf=12,learning_rate=.1,l2_regularization=0.)]
    if method in {'irf','crf'}:
        base=dict(n_estimators=16,max_depth=4,min_samples_leaf=12,max_thresholds=16)
        return [{**base,**(dict(penalty=v) if method=='irf' else dict(lambda1=v,lambda2=v))} for v in [0.,.05,.5]]
    if method=='maxrm':return [dict(n_estimators=16,max_depth=4,min_samples_leaf=12,robust=r) for r in [False,True]]
    if method=='meta':return [dict(iterations=3,trees_per_forest=4,max_depth=5,sampling_ratio=.7,alpha=-1.,beta=1.,use_mmd=w=='full',use_accuracy=w!='uniform',feature_masking=mask) for mask in [False,True] for w in ['uniform','accuracy','full']]
    if method in {'era','directional_era'}:return [dict(strength=s,max_iter=80,max_leaf_nodes=8,max_bins=32,min_samples_leaf=12,learning_rate=.1) for s in [0.,.1,1.]]
    if method=='anchor':return [dict(gamma=g,num_boost_round=100,learning_rate=.1,num_leaves=8,min_data_in_leaf=12) for g in [1.,2.,8.]]
    raise ValueError(method)


def candidates(method,task):
    result=[dict(parameters=p,reference=True) for p in references(method,task)]
    rng=np.random.default_rng(seed('space',method,task))
    def pick(values):return values[int(rng.integers(len(values)))]
    seen={digest(r['parameters']) for r in result}
    while len(result)<24:
        if method in {'rf','extra_trees'}:
            p=dict(n_estimators=pick([256,512]),max_depth=pick([None,8,16]),min_samples_leaf=pick([2,5,12,24]),max_features=pick(['sqrt',.5,1.]))
        elif method=='hist_gbdt':
            p=dict(max_iter=pick([200,400]),max_leaf_nodes=pick([8,15,31]),min_samples_leaf=pick([5,12,24,48]),learning_rate=pick([.03,.05,.1]),l2_regularization=pick([0.,1.,10.]))
        elif method in {'irf','crf'}:
            p=dict(n_estimators=pick([64,128]),max_depth=pick([4,6,8]),min_samples_leaf=pick([8,16,32]),max_thresholds=pick([16,32]))
            if method=='irf':p['penalty']=pick([0.,.001,.005,.01,.05,.1,.5,1.])
            else:p.update(lambda1=pick([0.,.001,.01,.1,.5]),lambda2=pick([0.,.001,.01,.1,.5]))
        elif method=='maxrm':p=dict(n_estimators=pick([32,64]),max_depth=pick([4,6,8]),min_samples_leaf=pick([12,24,48]),robust=pick([False,True]))
        elif method=='meta':p=dict(iterations=pick([3,5,8]),trees_per_forest=pick([8,16]),max_depth=pick([4,6,10]),sampling_ratio=pick([.5,.7,1.]),alpha=pick([0.,-.1,-1.,-3.]),beta=pick([0.,.5,1.,2.]),use_mmd=True,use_accuracy=True,feature_masking=pick([False,True]))
        elif method in {'era','directional_era'}:p=dict(strength=pick([0.,.03,.1,.3,1.]),max_iter=pick([160,320]),max_leaf_nodes=pick([8,15,31]),max_bins=pick([32,64]),min_samples_leaf=pick([12,24,48]),learning_rate=pick([.03,.05,.1]))
        elif method=='anchor':p=dict(gamma=pick([1.,1.5,2.,4.,8.,16.]),num_boost_round=pick([200,400]),learning_rate=pick([.03,.05,.1]),num_leaves=pick([8,15,31]),min_data_in_leaf=pick([12,24,48]))
        else:raise ValueError(method)
        h=digest(p)
        if h not in seen:seen.add(h);result.append(dict(parameters=p,reference=False))
    return [dict(candidate=i,**r) for i,r in enumerate(result)]


def screening_parameters(method,p):
    p=dict(p)
    # Only capacity/iterations reduced; every candidate retains full source rows.
    for key in ['n_estimators','max_iter','num_boost_round']:
        if key in p:p[key]=min(p[key],32 if key=='n_estimators' else 60)
    if method=='meta':p['iterations']=min(p['iterations'],3);p['trees_per_forest']=min(p['trees_per_forest'],4)
    return p


def final_jobs():
    result=[]
    for c in contexts():
        if c['dataset'] in {'beijing_pm25','appliances'}:continue
        for replica in range(REPLICAS if c['kind']=='synthetic' else 3):
            for model_seed in SEEDS:
                s={**c,'replica':5000+replica,'n':16384 if c['kind']=='synthetic' else 32000,'model_seed':model_seed,'data_replica':replica}
                s['job_id']=digest(s)[:16];result.append(s)
    return result
