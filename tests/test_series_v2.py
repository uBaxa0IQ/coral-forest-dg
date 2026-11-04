import numpy as np
import pytest
from nir_dg.series_v2.generators import allocation,draw,source_frame,target_frame
from nir_dg.series_v2.models import Model,EfficientMaxRM
from nir_dg.series_v2.design import contexts,jobs
from nir_dg.series_v2.data import source_real,features
from nir_dg.series_v2.engine import folds


def test_design_and_allocations():
    cs=contexts()
    assert len(cs)==381 and len(jobs())==762
    assert len({c["context_id"] for c in cs})==len(cs)
    for n,k,ratio in [(2048,16,8),(8192,8,8),(8192,16,1)]:
        counts=allocation(n,k,ratio,3)
        assert counts.sum()==n and counts.min()>=32
        assert counts.max()/counts.min()<=ratio*1.02


def test_dgp_prefix_and_pure_shifts():
    a,y=draw("regression","covariate",1000,0,np.random.default_rng(42))
    b,z=draw("regression","covariate",2000,2,np.random.default_rng(42))
    assert np.allclose(a[:,2:],b[:1000,2:])
    residual1=y-1.5*a[:,0]-.5*np.sin(a[:,1])
    residual2=z[:1000]-1.5*b[:1000,0]-.5*np.sin(b[:1000,1])
    assert np.allclose(residual1,residual2)
    # Same latent draws, changed prevalence; class-conditional offsets fixed.
    a,y=draw("classification","label_prior",10000,.1,np.random.default_rng(7))
    b,z=draw("classification","label_prior",10000,.9,np.random.default_rng(7))
    assert abs(y.mean()-.1)<.02 and abs(z.mean()-.9)<.02
    assert np.allclose(a[:,0]-.8*(2*y-1),b[:,0]-.8*(2*z-1))


def test_no_target_without_frozen_authorization():
    with pytest.raises((RuntimeError,FileNotFoundError)):
        target_frame({})


def test_matched_forest_members_and_structures():
    rng=np.random.default_rng(9);X=rng.normal(size=(240,5));y=(X[:,0]+rng.normal(size=240)>.2).astype(int);env=np.repeat(np.arange(4),60)
    a=Model("meta","classification",11,{"weighting":"uniform","masking":False}).fit(X,y,env)
    b=Model("meta","classification",11,{"weighting":"full","masking":False}).fit(X,y,env)
    for (fa,ca),(fb,cb) in zip(a.model_.members_,b.model_.members_):
        assert np.array_equal(ca,cb)
        assert np.array_equal(fa.predict_proba(X[:,ca]),fb.predict_proba(X[:,cb]))
    a=EfficientMaxRM(11,False,trees=2).fit(X,X[:,0],env)
    b=EfficientMaxRM(11,True,trees=2).fit(X,X[:,0],env)
    for ta,tb,ia,ib in zip(a.members_,b.members_,a.bootstrap_indices_,b.bootstrap_indices_):
        assert np.array_equal(ia,ib)
        assert np.array_equal(ta.tree_.feature,tb.tree_.feature)
        assert np.array_equal(ta.tree_.threshold,tb.tree_.threshold)
    assert all(v["after"]<=v["before"]+1e-5 for v in b.optimization_)


def test_source_sizes_and_temporal_embargo():
    for spec in [s for s in contexts() if s["kind"]=="real" and s["replica"]==0]:
        f=source_real(spec)
        assert len(f)==spec["n"] and f.__env.nunique()==spec["k"]
        assert not set(features(f)) & {"ST","__env","__y","__time"}
        if spec["dataset"]=="appliances":
            for tr,va in folds(f,spec):
                assert f.iloc[tr].__time.max()<f.iloc[va].__time.min()-24*3600*10**9


def test_regression_zero_penalty_has_same_admissibility():
    from nir_dg.series_v2.controlled_trees import DomainForest
    X=np.arange(80,dtype=float)[:,None];y=np.sin(X[:,0]);env=np.repeat([0,1],40)
    # The feature supports of the domains do not overlap: no candidate can put
    # each active domain on BOTH sides, even in the zero-penalty control.
    for penalty in [0.,1.]:
        m=DomainForest(task="regression",penalty=penalty,n_estimators=2).fit(X,y,env)
        assert all(t.feature==-1 for t in m.trees_)
