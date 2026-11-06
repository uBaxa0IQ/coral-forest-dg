"""Source-only component ablations of the labelled Meta-forests interpretation."""
import json
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits
from nir_dg.datasets import ROOT, load_source
from nir_dg.experiment import append_record
from nir_dg.meta_forests import MetaForestsClassifier
from nir_dg.metrics import score
from nir_dg.preprocessing import make_preprocessor
from nir_dg.protocol import code_fingerprint, source_folds

if __name__ == "__main__":
    out = ROOT / "runs/meta_ablations_v1"
    out.mkdir(exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(dict(code_hash=code_fingerprint(ROOT), target_access=False, status="independent_interpretation", datasets=["acs_income", "synthetic_classification"]), indent=2), encoding="utf-8")
    variants = {"forest_only": (False, False, False), "meta_validation_only": (True, False, False), "mmd_only": (False, True, False), "masking_only": (False, False, True), "full": (True, True, True)}
    for name in ["acs_income", "synthetic_classification"]:
        data = load_source(name)
        for fold, train, valid in source_folds(data, 2):
            prep = make_preprocessor(data.X.iloc[train])
            Xt, Xv = prep.fit_transform(data.X.iloc[train]), None
            Xv = prep.transform(data.X.iloc[valid])
            for seed in [11, 23, 47]:
                for variant, (accuracy, mmd, masking) in variants.items():
                    start = perf_counter()
                    with threadpool_limits(1):
                        model = MetaForestsClassifier(use_accuracy=accuracy, use_mmd=mmd, feature_masking=masking, random_state=seed).fit(Xt, data.y[train], data.environments[train])
                        metrics = score("classification", data.y[valid], model.predict_proba(Xv)[:, 1])
                    append_record(out / "source_cv_metrics.csv", dict(dataset=name, fold=fold, seed=seed, variant=variant, roc_auc=metrics["roc_auc"], fit_predict_seconds=perf_counter() - start, forests=len(model.members_), total_trees=sum(h["n_trees"] for h in model.history_), implementation_status="independent_interpretation"))
        print(name, "ablations complete", flush=True)
