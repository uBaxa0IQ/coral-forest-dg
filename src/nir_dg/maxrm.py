"""MaxRM-RF-posthoc, MSE objective, independent sklearn/CVXPY implementation."""
import numpy as np
import cvxpy as cp
from sklearn.tree import DecisionTreeRegressor


class MaxRMForestRegressor:
    implementation_status = "independent_reimplementation"

    def __init__(self, n_estimators=12, max_depth=4, min_samples_leaf=12, reoptimize=True, random_state=11):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_samples_leaf, self.reoptimize, self.random_state = min_samples_leaf, reoptimize, random_state

    def fit(self, X, y, environments):
        X, y, env = np.asarray(X), np.asarray(y, dtype=float), np.asarray(environments)
        rng = np.random.default_rng(self.random_state)
        self.trees_, self.optimization_, self.bootstrap_indices_ = [], [], []
        self.y_mean_, self.y_scale_ = y.mean(), max(y.std(), 1e-8)
        z = (y - self.y_mean_) / self.y_scale_
        for _ in range(self.n_estimators):
            ids = rng.integers(len(y), size=len(y))
            self.bootstrap_indices_.append(ids)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=1., random_state=int(rng.integers(2**31 - 1)))
            tree.fit(X[ids], z[ids])
            if self.reoptimize:
                leaves, inverse = np.unique(tree.apply(X[ids]), return_inverse=True)
                theta = cp.Variable(len(leaves))
                worst = cp.Variable(nonneg=True)
                constraints = [cp.sum_squares(theta[inverse[env[ids] == e]] - z[ids][env[ids] == e]) / np.sum(env[ids] == e) <= worst for e in np.unique(env[ids])]
                problem = cp.Problem(cp.Minimize(worst), constraints)
                problem.solve(solver="CLARABEL")
                if problem.status != cp.OPTIMAL or theta.value is None:
                    raise RuntimeError(f"MaxRM solver did not return a verified optimum: {problem.status}")
                before = max(np.mean((tree.predict(X[ids][env[ids] == e]) - z[ids][env[ids] == e]) ** 2) for e in np.unique(env[ids]))
                after = max(np.mean((theta.value[inverse[env[ids] == e]] - z[ids][env[ids] == e]) ** 2) for e in np.unique(env[ids]))
                if after > before + 1e-5 * max(1, before):
                    raise RuntimeError("MaxRM solution violates no-worse-than-feasible-baseline invariant")
                for leaf, value in zip(leaves, theta.value):
                    tree.tree_.value[leaf, 0, 0] = value
                self.optimization_.append(dict(status=problem.status, before=float(before), after=float(after)))
            self.trees_.append(tree)
        return self

    def predict(self, X):
        return self.y_mean_ + self.y_scale_ * np.mean([t.predict(X) for t in self.trees_], axis=0)
