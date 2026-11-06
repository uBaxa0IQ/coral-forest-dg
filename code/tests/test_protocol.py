import json
import numpy as np
import pandas as pd
import pytest
from nir_dg.protocol import ProtocolError, SourceData, TargetVault, freeze_protocol, source_folds
from nir_dg.preprocessing import make_preprocessor
from nir_dg.metrics import score


def source(temporal=False):
    return SourceData(pd.DataFrame({"a": np.arange(12.)}), np.arange(12.) % 2, np.repeat([0, 1, 2, 3], 3), np.arange(12), "classification", "test", temporal, np.arange(12) if temporal else None)


def design():
    return dict(datasets=["d"], methods=["rf"], seeds=[11], metrics=["roc_auc"], hpo={"rf": [{}]}, source_folds=["a"], target_ids=["held_out"], source_fingerprints={"d": "abc"}, experiment_manifest={"test": {"target_ids": ["held_out"]}})


def test_target_loader_is_not_called_without_freeze(tmp_path):
    calls = []
    vault = TargetVault(tmp_path, tmp_path, lambda entry: calls.append(entry))
    with pytest.raises(ProtocolError, match="locked"):
        vault.load_for_evaluation("test")
    assert calls == []


def test_freeze_integrity_and_single_access(tmp_path):
    freeze_protocol(tmp_path, design(), tmp_path)
    calls = []
    vault = TargetVault(tmp_path, tmp_path, lambda entry: calls.append(entry))
    with pytest.raises(ProtocolError, match="not preregistered"):
        vault.load_for_evaluation("unknown")
    vault.load_for_evaluation("test")
    with pytest.raises(ProtocolError, match="already accessed"):
        vault.load_for_evaluation("test")
    assert len(calls) == 1
    path = tmp_path / "frozen_protocol.json"
    value = json.loads(path.read_text())
    value["design"]["seeds"] = [999]
    path.write_text(json.dumps(value))
    with pytest.raises(ProtocolError, match="modified"):
        vault.load_for_evaluation("test")


def test_code_changes_invalidate_freeze(tmp_path):
    (tmp_path / "src").mkdir()
    path = tmp_path / "src/model.py"
    path.write_text("v = 1")
    freeze_protocol(tmp_path, design(), tmp_path)
    path.write_text("v = 2")
    with pytest.raises(ProtocolError, match="code changed"):
        TargetVault(tmp_path, tmp_path, lambda _: None).load_for_evaluation("test")


def test_domain_disjoint_and_temporal_order():
    for temporal in (False, True):
        d = source(temporal)
        for _, train, valid in source_folds(d):
            assert not set(d.environments[train]) & set(d.environments[valid])
            if temporal:
                assert d.time_ids[train].max() < d.time_ids[valid].min()


def test_preprocessor_does_not_fit_held_out_statistics():
    d = source()
    p = make_preprocessor(d.X.iloc[:6])
    p.fit(d.X.iloc[:6])
    assert p.named_transformers_["numeric"].named_steps["scale"].mean_[0] == 2.5
    changed = pd.DataFrame({"a": [np.nan, 1e9]})
    p.transform(changed)
    assert p.named_transformers_["numeric"].named_steps["scale"].mean_[0] == 2.5


def test_single_class_auc_is_missing_not_replaced():
    assert np.isnan(score("classification", [0, 0], [.1, .2])["roc_auc"])
