import numpy as np
import pytest
from nir_dg.trees import DomainForest, crf_node_loss, entropy, irf_classification_penalty, irf_regression_penalty, mutual_information
from nir_dg.meta_forests import MetaForestsClassifier, mmd_squared, normalized_weights
from nir_dg.maxrm import MaxRMForestRegressor
from nir_dg.models import Adapter, applicable_methods


def toy():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(240, 4))
    env = np.repeat(np.arange(4), 60)
    y = 2 * X[:, 0] + env + rng.normal(size=240)
    return X, y, env


def test_penalties_and_information_identities():
    assert entropy([1, 1]) == pytest.approx(1)
    assert mutual_information([[10, 0], [0, 10]]) == pytest.approx(1)
    assert mutual_information([[5, 5], [5, 5]]) == pytest.approx(0)
    assert irf_classification_penalty([[5, 2], [5, 2]], [[10, 10], [10, 10]]) == 0
    assert irf_classification_penalty([[5, 2], [2, 5]], [[10, 10], [10, 10]]) > 0
    assert irf_regression_penalty([2, 4, 6], [1, 3, 5]) == 0
    assert irf_regression_penalty([2, 4, 9], [1, 3, 5]) > 0
    assert crf_node_loss(np.array([[5, 5], [5, 5]]), 1, 1) == pytest.approx(0)


def test_irf_domainwise_bootstrap_and_reproducibility():
    X, y, env = toy()
    a = DomainForest(task="regression", n_estimators=3).fit(X, y, env)
    b = DomainForest(task="regression", n_estimators=3).fit(X, y, env)
    for ids in a.bootstrap_indices_:
        assert np.array_equal(np.bincount(env[ids]), np.bincount(env))
    np.testing.assert_allclose(a.predict(X), b.predict(X))


def test_meta_mmd_weights_source_meta_test_and_masks():
    X, y, env = toy()
    assert mmd_squared(X, X, .25) == pytest.approx(0, abs=1e-12)
    assert mmd_squared(X, X + 10, .25) > 0
    np.testing.assert_allclose(normalized_weights([1000, 1000]), [.5, .5])
    model = MetaForestsClassifier(iterations=3, trees_per_forest=2).fit(X, (y > 1).astype(int), env)
    assert len(model.members_) == 3 * (len(np.unique(env)) - 1)
    assert model.weights_.sum() == pytest.approx(1)
    assert (model.weights_ >= 0).all()
    for h in model.history_:
        assert h["train_domain"] != h["meta_test_domain"]
        assert int(h["meta_test_domain"]) in env
        assert h["features"]
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.)
    again = MetaForestsClassifier(iterations=3, trees_per_forest=2).fit(X, (y > 1).astype(int), env)
    np.testing.assert_allclose(model.predict_proba(X), again.predict_proba(X))


def test_maxrm_is_minimax_not_pooled_and_preserves_structure():
    X, y, env = toy()
    model = MaxRMForestRegressor(n_estimators=2).fit(X, y, env)
    baseline = MaxRMForestRegressor(n_estimators=2, reoptimize=False).fit(X, y, env)
    for row in model.optimization_:
        assert row["after"] <= row["before"] + 1e-5
    for a, b in zip(model.trees_, baseline.trees_):
        np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
        np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
    assert not np.allclose(model.predict(X), baseline.predict(X))


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_all_registered_methods_fit_predict(task):
    X, y, env = toy()
    if task == "classification":
        y = (y > 1).astype(int)
    for name in applicable_methods(task):
        model = Adapter(name, task, 11).fit(X, y, env)
        pred = model.predict_values(X[:10])
        assert pred.shape == (10,), name
        assert np.isfinite(pred).all(), name


def test_era_baseline_matches_hist_engine_on_simple_source_data():
    # Same engine with DG components explicitly disabled. Compare to standard
    # HistGBDT only on unambiguous one-feature non-tied data; no universal equivalence claim.
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.arange(240, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0] / 20)
    env = np.tile(np.arange(4), 60)
    era = Adapter("era_paired", "regression", 11).fit(X, y, env)
    ref = HistGradientBoostingRegressor(max_iter=30, max_leaf_nodes=8, max_bins=16, min_samples_leaf=12, early_stopping=False, random_state=11, learning_rate=.1).fit(X, y)
    np.testing.assert_allclose(era.predict_values(X), ref.predict(X), atol=1e-8)
