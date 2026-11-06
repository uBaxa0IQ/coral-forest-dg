"""Full, explicitly labelled interpretation of the underspecified Meta-forests algorithm."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel


def mmd_squared(X, Y, gamma):
    """Biased nonnegative RBF MMD^2; bandwidth is set using training sources only."""
    return max(0., float(rbf_kernel(X, X, gamma=gamma).mean() + rbf_kernel(Y, Y, gamma=gamma).mean() - 2 * rbf_kernel(X, Y, gamma=gamma).mean()))


def normalized_weights(log_weights):
    values = np.asarray(log_weights, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite forest log weights")
    values = np.exp(values - values.max())
    return values / values.sum()


class MetaForestsClassifier:
    implementation_status = "independent_interpretation"
    interpretation = "positive_monotone_update_rbf_previous_feature_mask"

    def __init__(self, iterations=3, trees_per_forest=4, max_depth=4, sampling_ratio=.7,
                 alpha=-1., beta=1., use_mmd=True, use_accuracy=True, feature_masking=True,
                 random_state=11):
        self.iterations, self.trees_per_forest, self.max_depth = iterations, trees_per_forest, max_depth
        self.sampling_ratio, self.alpha, self.beta = sampling_ratio, alpha, beta
        self.use_mmd, self.use_accuracy, self.feature_masking = use_mmd, use_accuracy, feature_masking
        self.random_state = random_state

    def fit(self, X, y, environments):
        X, y, environments = np.asarray(X), np.asarray(y), np.asarray(environments)
        domains = np.unique(environments)
        if len(domains) < 3:
            raise ValueError("Meta-forests requires at least three source domains in this implementation")
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.members_, self.history_, logs = [], [], []
        self.last_importances_ = {}
        self.source_domains_ = domains.copy()
        previous_weights, used_features, distances = {}, {}, []
        gamma = 1. / X.shape[1]
        for iteration in range(self.iterations):
            heldout = rng.choice(domains)
            valid_ids = np.flatnonzero(environments == heldout)
            valid_ids = rng.choice(valid_ids, min(len(valid_ids), 256), replace=False)
            for domain in domains[domains != heldout]:
                train = np.flatnonzero(environments == domain)
                chosen = rng.choice(train, max(2, int(self.sampling_ratio * len(train))), replace=True)
                available = np.arange(X.shape[1])
                if self.feature_masking and domain in used_features:
                    available = np.setdiff1d(available, used_features[domain])
                    if not len(available):
                        # Published masking policy does not define this case. Explicit
                        # interpretation: keep one least-important previous feature.
                        available = np.array([self.last_importances_[domain].argmin()])
                forest = RandomForestClassifier(n_estimators=self.trees_per_forest, max_depth=self.max_depth, min_samples_leaf=5, max_features="sqrt", n_jobs=1, random_state=int(rng.integers(2**31 - 1)))
                forest.fit(X[chosen][:, available], y[chosen])
                accuracy = accuracy_score(y[valid_ids], forest.predict(X[valid_ids][:, available]))
                distance = mmd_squared(X[chosen[:256]][:, available], X[valid_ids][:, available], gamma)
                baseline_distance = np.mean(distances) if distances else 0.
                # Paper Eq. 3 has log-domain/sign/initial-index ambiguities. This
                # monotone positive update is deliberately NOT called an exact replica.
                log_weight = np.log(previous_weights.get(domain, 1. / (len(domains) - 1)))
                if self.use_mmd:
                    log_weight += self.alpha * (distance - baseline_distance)
                if self.use_accuracy:
                    positive_accuracy = max(np.exp(accuracy) - np.exp(1. / len(self.classes_)), 1e-8)
                    log_weight += self.beta * np.log(positive_accuracy)
                logs.append(log_weight)
                self.members_.append((forest, available))
                self.history_.append(dict(iteration=iteration, train_domain=str(domain), meta_test_domain=str(heldout), accuracy=float(accuracy), mmd_squared=distance, features=available.tolist(), n_trees=self.trees_per_forest))
                previous_weights[domain] = float(np.exp(np.clip(log_weight, -600, 600)))
                distances.append(distance)
                used = np.unique(np.concatenate([t.tree_.feature[t.tree_.feature >= 0] for t in forest.estimators_]))
                used_features[domain] = available[used]
                if not hasattr(self, "last_importances_"):
                    self.last_importances_ = {}
                importance = np.zeros(X.shape[1])
                importance[available] = forest.feature_importances_
                self.last_importances_[domain] = importance
        self.weights_ = normalized_weights(logs)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        out = np.zeros((len(X), len(self.classes_)))
        for weight, (forest, features) in zip(self.weights_, self.members_):
            # A bootstrap from one source may omit a class: map columns explicitly.
            p = forest.predict_proba(X[:, features])
            for j, cls in enumerate(forest.classes_):
                out[:, np.flatnonzero(self.classes_ == cls)[0]] += weight * p[:, j]
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
