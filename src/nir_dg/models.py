from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from nir_dg.trees import DomainForest
from nir_dg.meta_forests import MetaForestsClassifier
from nir_dg.maxrm import MaxRMForestRegressor


@dataclass(frozen=True)
class MethodSpec:
    family: str
    fidelity: str
    paired_baseline: str


METHODS = {
    "rf": MethodSpec("RF", "baseline_only", ""),
    "hist_gbdt": MethodSpec("GBDT", "baseline_only", ""),
    "irf": MethodSpec("RF", "independent_reimplementation", "irf_paired"),
    "irf_paired": MethodSpec("RF", "baseline_only", ""),
    "crf": MethodSpec("RF", "independent_reimplementation", "crf_paired"),
    "crf_paired": MethodSpec("RF", "baseline_only", ""),
    "meta_forests_interpretation": MethodSpec("RF", "independent_interpretation", "meta_matched_rf"),
    "meta_matched_rf": MethodSpec("RF", "baseline_only", ""),
    "maxrm_posthoc": MethodSpec("RF", "independent_reimplementation", "maxrm_paired"),
    "maxrm_paired": MethodSpec("RF", "baseline_only", ""),
    "era": MethodSpec("GBDT", "exact_author_code", "era_paired"),
    "directional_era": MethodSpec("GBDT", "exact_author_code", "era_paired"),
    "era_paired": MethodSpec("GBDT", "baseline_only", ""),
    "anchor": MethodSpec("GBDT", "exact_author_code", "anchor_paired"),
    "anchor_paired": MethodSpec("GBDT", "baseline_only", ""),
}


class Adapter:
    def __init__(self, name, task, seed, params=None):
        self.name, self.task, self.seed = name, task, seed
        self.params = dict(params or {})

    def fit(self, X, y, environments):
        name, task, seed, p = self.name, self.task, self.seed, self.params
        if name in {"rf", "meta_matched_rf"}:
            n = 3 * (len(np.unique(environments)) - 1) * 4 if name == "meta_matched_rf" else p.get("n_estimators", 12)
            cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
            self.model = cls(n_estimators=n, max_depth=p.get("max_depth", 4), min_samples_leaf=12, random_state=seed, n_jobs=1)
            self.model.fit(X, y)
        elif name == "hist_gbdt":
            cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
            self.model = cls(max_iter=p.get("rounds", 30), max_leaf_nodes=8, min_samples_leaf=12, early_stopping=False, random_state=seed, learning_rate=p.get("learning_rate", .1))
            self.model.fit(X, y)
        elif name in {"irf", "irf_paired", "crf", "crf_paired"}:
            self.model = DomainForest(task=task, mechanism="crf" if name.startswith("crf") else "irf", penalty=p.get("penalty", 1.) if name == "irf" else 0., lambda1=p.get("lambda1", 1.) if name == "crf" else 0., lambda2=p.get("lambda2", 1.) if name == "crf" else 0., random_state=seed)
            self.model.fit(X, y, environments)
        elif name == "meta_forests_interpretation":
            self.model = MetaForestsClassifier(random_state=seed, **p).fit(X, y, environments)
        elif name in {"maxrm_posthoc", "maxrm_paired"}:
            if task != "regression":
                raise ValueError("MaxRM regression only")
            self.model = MaxRMForestRegressor(reoptimize=name == "maxrm_posthoc", random_state=seed, **p).fit(X, y, environments)
        elif name in {"era", "directional_era", "era_paired"}:
            if task != "regression":
                raise ValueError("Era classifier API not verified")
            from erasplit.ensemble import EraHistGradientBoostingRegressor
            weights = {"era": (1, 0, 0), "directional_era": (0, 1, 0), "era_paired": (0, 0, 1)}[name]
            era_params = {k: v for k, v in p.items() if k != "rounds"}
            self.model = EraHistGradientBoostingRegressor(gamma=weights[0], blama=weights[1], vanna=weights[2], max_iter=p.get("rounds", 30), max_leaf_nodes=8, max_bins=16, min_samples_leaf=12, early_stopping=False, random_state=seed, n_jobs=1, learning_rate=.1, **era_params)
            _, env = np.unique(environments, return_inverse=True)
            self.model.fit(X, y, env.astype(np.int32))
        elif name in {"anchor", "anchor_paired"}:
            from anchorboosting import AnchorBooster
            gamma = p.get("gamma", 4.) if name == "anchor" else 1.
            self.model = AnchorBooster(gamma=gamma, num_boost_round=p.get("rounds", 30), objective="binary" if task == "classification" else "regression", alpha_cvar=1., learning_rate=.1, num_leaves=8, min_data_in_leaf=12, min_gain_to_split=1e-8, verbosity=-1, num_threads=1, seed=seed, deterministic=True, force_col_wise=True)
            _, env = np.unique(environments, return_inverse=True)
            anchors = np.eye(env.max() + 1)[env]
            self.model.fit(X, y, Z=anchors)
        else:
            raise ValueError(f"Unknown method {name}")
        return self

    def predict_values(self, X):
        if self.task == "classification" and not self.name.startswith("anchor"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


def applicable_methods(task):
    shared = ["rf", "hist_gbdt", "irf_paired", "irf", "anchor_paired", "anchor"]
    if task == "classification":
        return shared + ["crf_paired", "crf", "meta_matched_rf", "meta_forests_interpretation"]
    return shared + ["maxrm_paired", "maxrm_posthoc", "era_paired", "era", "directional_era"]
