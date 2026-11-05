import numpy as np
import pandas as pd
import pytest
from research_versions.robustness_v3.controlled_trees import DomainForest,safe_threshold
from research_versions.robustness_v3.preprocessing import make_preprocessor,canonical_frame


def test_adjacent_float_partition_and_extreme_endpoints():
    lower=np.nextafter(1.,np.inf);upper=np.nextafter(lower,np.inf)
    assert lower+(upper-lower)/2==upper  # reproduces the old rounded-up midpoint
    assert lower<=safe_threshold(lower,upper)<upper
    assert np.isfinite(safe_threshold(-1e308,1e308))
    X=np.tile(np.r_[np.full(32,lower),np.full(32,upper)],4)[:,None]
    y=np.tile(np.r_[np.zeros(32),np.ones(32)],4);env=np.repeat(np.arange(4),64)
    for task in ["classification","regression"]:
        model=DomainForest(task=task,n_estimators=2,penalty=0,random_state=11).fit(X,y,env)
        assert all(tree.feature==0 for tree in model.trees_)
        assert np.isfinite(model.predict_values(X)).all()
        assert np.array_equal(model.predict_values(X),y)


def test_cache_roundtrip_and_nullable_scalars_are_equivalent(tmp_path):
    raw=pd.DataFrame({"number":[1.,2.,3.,np.nan],"wind":["N","S",np.nan,"N"]})
    path=tmp_path/"source.parquet";raw.to_parquet(path,index=False)
    cached=pd.read_parquet(path)
    processor=make_preprocessor(raw)
    a=processor.fit_transform(raw);b=processor.transform(cached)
    assert np.array_equal(a,b)
    other=raw.copy();other.loc[2,"wind"]=pd.NA
    assert np.array_equal(a,processor.transform(other))
    assert canonical_frame(pd.DataFrame({"n":pd.Series([1,pd.NA],dtype="Int64")})).n.dtype==np.float64


def test_bad_split_endpoints_rejected():
    for pair in [(1,1),(2,1),(np.nan,2),(1,np.inf)]:
        with pytest.raises(ValueError):
            safe_threshold(*pair)


def test_centered_maxrm_retains_paired_structure_and_certifies_objective():
    from research_versions.robustness_v3.maxrm import CertifiedMaxRM
    rng=np.random.default_rng(621)
    X=rng.normal(size=(240,3));env=np.repeat(np.arange(4),60)
    y=X[:,0]+rng.normal(size=240)*(1+env)
    control=CertifiedMaxRM(17,False,n_estimators=2).fit(X,y,env)
    robust=CertifiedMaxRM(17,True,n_estimators=2).fit(X,y,env)
    for a,b,ia,ib,c in zip(control.members_,robust.members_,control.bootstrap_indices_,robust.bootstrap_indices_,robust.certificates_):
        assert np.array_equal(ia,ib)
        assert np.array_equal(a.tree_.children_left,b.tree_.children_left)
        assert np.array_equal(a.tree_.threshold,b.tree_.threshold)
        actual=max(np.mean((b.predict(X[ib][env[ib]==e])-y[ib][env[ib]==e])**2) for e in np.unique(env))
        assert np.isclose(actual,c['primal'],rtol=0,atol=1e-10)
        assert -1e-6<=c['gap']<=1e-6*max(1,actual)
