import json
import numpy as np
import pandas as pd
import pytest
from research_versions.tuned_v4.design import candidates,methods,digest,seed,contexts,final_jobs
from research_versions.tuned_v4.models import TunedModel,control_parameters
from research_versions.tuned_v4 import data
from research_versions.tuned_v4.engine import metrics


def test_spaces_and_source_domain_validation_are_fixed_and_disjoint():
    for task in ['classification','regression']:
        for method in methods(task):
            grid=candidates(method,task)
            assert len(grid)==24 and len({digest(c['parameters']) for c in grid})==24
            assert any(c['reference'] for c in grid)
            assert grid==candidates(method,task)
    f=pd.DataFrame({'x':np.arange(80),'__y':np.arange(80)%2,'__env':np.repeat(np.arange(8),10),'__row_id':np.arange(80).astype(str)})
    splits=list(data.folds(f,dict(dataset='synthetic')))
    assert len(splits)==4
    assert len({f.iloc[va].__env.iloc[0] for _,va in splits})==4
    for tr,va in splits:assert not set(f.iloc[tr].__env)&set(f.iloc[va].__env)
    assert len(final_jobs())==272
    assert seed('sealed_v4_target','classification','iid',0,0)!=seed('source','classification','iid',0,0)


def test_new_target_is_closed_without_freeze_and_rejects_unregistered_job(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'RUN',tmp_path)
    with pytest.raises(RuntimeError,match='sealed'):data.synthetic_target({})
    body=dict(jobs=[]);(tmp_path/'final_protocol.json').write_text(json.dumps(dict(protocol=body,sha256=digest(body))))
    with pytest.raises(RuntimeError,match='Unregistered'):data.synthetic_target({'test':'unit test, no scientific target generated'})


@pytest.mark.parametrize('task',['classification','regression'])
def test_adapters_fit_and_mechanistic_controls_retain_capacity(task):
    rng=np.random.default_rng(761);X=rng.normal(size=(320,4));env=np.repeat(np.arange(8),40)
    y=(X[:,0]>0).astype(int) if task=='classification' else X[:,0]+rng.normal(size=320)
    for method in methods(task):
        p=candidates(method,task)[0]['parameters'].copy()
        for key in ['n_estimators','max_iter','num_boost_round','trees_per_forest']:
            if key in p:p[key]=2
        if 'iterations' in p:p['iterations']=3
        model=TunedModel(method,task,11,p).fit(X,y,env)
        assert np.isfinite(model.predict(X)).all(),method
        if method=='meta':
            control=model.uniform_control()
            assert control.model_.members_ is model.model_.members_
            assert np.allclose(control.model_.weights_,1/len(control.model_.members_))
        else:
            cp=control_parameters(method,p)
            if cp is not None:
                assert {k:v for k,v in p.items() if k not in ['penalty','lambda1','lambda2','robust','strength','gamma']}=={k:v for k,v in cp.items() if k not in ['penalty','lambda1','lambda2','robust','strength','gamma']}


def test_primary_metrics_independently_handle_ties_and_scale():
    assert metrics('classification',np.array([0,1,0,1]),np.array([.1,.5,.5,.9]))['roc_auc']==.875
    assert metrics('regression',np.array([1.,3.]),np.array([0.,1.]))['mse']==2.5
