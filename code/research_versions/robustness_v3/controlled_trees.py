"""Prospective numerical correction of v2; source-tested, not target-validated."""
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator


def safe_threshold(lower, upper):
    """A finite threshold realizing the scored sorted partition, including ULP ties."""
    lower,upper=float(lower),float(upper)
    if not np.isfinite([lower,upper]).all() or not lower<upper:
        raise ValueError("Expected two ordered finite distinct split endpoints")
    with np.errstate(over="ignore",invalid="ignore"):
        mid=lower+(upper-lower)/2
    if not np.isfinite(mid):
        mid=lower/2+upper/2
    return max(lower,min(mid,float(np.nextafter(upper,lower))))


def entropy(counts, axis=-1):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=axis, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    return -(p * np.log2(np.maximum(p, np.finfo(float).tiny))).sum(axis=axis)


def mutual_information(joint):
    joint = np.asarray(joint, dtype=float)
    return float(entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0)) - entropy(joint.ravel()))


def irf_classification_penalty(left_counts, parent_counts):
    """AAAI 2024 Eqs. 16-19, columns ordered (class 0, class 1)."""
    cr = (np.asarray(left_counts, dtype=float) + .5) / (np.asarray(parent_counts, dtype=float) + 1.)
    invariant = cr[..., 1] / cr[..., 0]
    return invariant.max(axis=-1) / invariant.min(axis=-1) - 1.


def irf_regression_penalty(left_means, parent_means):
    """AAAI 2024 Eqs. 20-22, population variance across active source domains."""
    return np.var(np.asarray(left_means) - np.asarray(parent_means), axis=-1)


def crf_node_loss(joint, lambda1, lambda2):
    # Joint cells are environment x class. This is the decomposable paper objective.
    return float(entropy(joint.sum(axis=0)) + lambda1 * mutual_information(joint) - lambda2 * entropy(joint.sum(axis=1)))


def crf_losses(joints, lambda1, lambda2):
    hy = entropy(joints.sum(axis=1))
    hz = entropy(joints.sum(axis=2))
    h_joint = entropy(joints.reshape(len(joints), -1))
    return hy + lambda1 * (hy + hz - h_joint) - lambda2 * hz


@dataclass
class Node:
    value: float
    feature: int = -1
    threshold: float = 0.
    left: object = None
    right: object = None


class DomainForest(BaseEstimator):
    """IRF (both tasks) or paper-objective CRF (binary classification).

    Candidate thresholds are a documented source-node quantile grid for bounded
    compute. It is an independent implementation, not a verified port of R code.
    """
    def __init__(self, task="classification", mechanism="irf", n_estimators=16, max_depth=4,
                 min_samples_leaf=12, penalty=1., lambda1=1., lambda2=1., max_thresholds=16,
                 random_state=11):
        self.task, self.mechanism = task, mechanism
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_samples_leaf, self.penalty = min_samples_leaf, penalty
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.max_thresholds, self.random_state = max_thresholds, random_state

    def fit(self, X, y, environments):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        environments, env = np.unique(environments, return_inverse=True)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Forest requires finite fold-preprocessed inputs")
        if self.mechanism == "crf" and self.task != "classification":
            raise ValueError("CRF regression is not implemented or scientifically verified")
        if self.task == "classification" and not np.isin(y, [0, 1]).all():
            raise ValueError("Only binary classification verified in this implementation")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.trees_, self.bootstrap_indices_ = [], []
        for _ in range(self.n_estimators):
            if self.mechanism == "irf":
                indices = np.concatenate([rng.choice(ids := np.flatnonzero(env == e), len(ids), replace=True) for e in range(len(environments))])
            else:
                indices = rng.integers(len(y), size=len(y))
            self.bootstrap_indices_.append(indices)
            self.trees_.append(self._grow(X[indices], y[indices], env[indices], 0))
        return self

    def _grow(self, X, y, env, depth):
        if not len(y):
            raise RuntimeError("Empty node violates the corrected partition invariant")
        node = Node(float(y.mean()))
        n = len(y)
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or np.ptp(y) == 0:
            return node
        _, env = np.unique(env, return_inverse=True)
        k = env.max() + 1
        parent_n = np.bincount(env, minlength=k)
        parent_y = np.bincount(env, weights=y, minlength=k)
        parent_joint = np.column_stack([parent_n - parent_y, parent_y])
        if self.mechanism == "crf":
            parent_loss = crf_node_loss(parent_joint, self.lambda1, self.lambda2)
        elif self.task == "classification":
            parent_loss = 2 * y.mean() * (1 - y.mean())
        else:
            parent_loss = np.var(y)
        best = None
        # v2: both the regularized model and zero-penalty control must improve
        # their parent objective. This is a registered variant, not author code.
        best_loss = parent_loss - 1e-12
        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind="stable")
            x, z, e = X[order, feature], y[order], env[order]
            positions = np.flatnonzero(x[:-1] < x[1:]) + 1
            positions = positions[(positions >= self.min_samples_leaf) & (positions <= n - self.min_samples_leaf)]
            if len(positions) > self.max_thresholds:
                positions = positions[np.unique(np.linspace(0, len(positions) - 1, self.max_thresholds).astype(int))]
            if not len(positions):
                continue
            indicator = np.eye(k)[e]
            count = np.cumsum(indicator, axis=0)[positions - 1]
            sum_y = np.cumsum(indicator * z[:, None], axis=0)[positions - 1]
            right_count, right_sum = parent_n - count, parent_y - sum_y
            w = positions / n
            if self.mechanism == "crf":
                left_joints = np.stack([count - sum_y, sum_y], axis=-1)
                right_joints = np.stack([right_count - right_sum, right_sum], axis=-1)
                losses = w * crf_losses(left_joints, self.lambda1, self.lambda2) + (1 - w) * crf_losses(right_joints, self.lambda1, self.lambda2)
            elif self.task == "classification":
                p_left, p_right = sum_y.sum(axis=1) / positions, right_sum.sum(axis=1) / (n - positions)
                losses = w * 2 * p_left * (1 - p_left) + (1 - w) * 2 * p_right * (1 - p_right)
                if self.penalty:
                    losses += self.penalty * irf_classification_penalty(np.stack([count - sum_y, sum_y], axis=-1), parent_joint)
            else:
                cumulative_sq = np.cumsum(z * z)[positions - 1]
                left_sum = sum_y.sum(axis=1)
                right_sum_total = right_sum.sum(axis=1)
                losses = (cumulative_sq - left_sum ** 2 / positions + (z @ z - cumulative_sq) - right_sum_total ** 2 / (n - positions)) / n
                # Identical candidate admissibility at EVERY penalty, including 0.
                complete = (count > 0).all(axis=1) & (right_count > 0).all(axis=1)
                if self.penalty:
                    means = sum_y / np.maximum(count, 1)
                    losses += self.penalty * irf_regression_penalty(means, parent_y / parent_n)
                losses[~complete] = np.inf
            index = int(np.argmin(losses))
            if losses[index] < best_loss:
                best_loss = float(losses[index])
                pos = positions[index]
                best = feature, safe_threshold(x[pos-1],x[pos])
        if best is None:
            return node
        feature, threshold = best
        left = X[:, feature] <= threshold
        if left.sum()<self.min_samples_leaf or (~left).sum()<self.min_samples_leaf:
            raise RuntimeError("Scored and realized partitions disagree on leaf support")
        if self.task=="regression" and self.mechanism=="irf":
            active=set(env)
            if set(env[left])!=active or set(env[~left])!=active:
                raise RuntimeError("Realized partition violates shared domain admissibility")
        node.feature, node.threshold = feature, float(threshold)
        node.left = self._grow(X[left], y[left], env[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], env[~left], depth + 1)
        return node

    @staticmethod
    def _predict_tree(node, X, result, ids):
        if node.feature < 0:
            result[ids] = node.value
            return
        mask = X[ids, node.feature] <= node.threshold
        DomainForest._predict_tree(node.left, X, result, ids[mask])
        DomainForest._predict_tree(node.right, X, result, ids[~mask])

    def predict(self, X):
        p = self.predict_values(X)
        return (p >= .5).astype(int) if self.task == "classification" else p

    def predict_values(self, X):
        X = np.asarray(X, dtype=float)
        results = []
        for tree in self.trees_:
            values = np.empty(len(X))
            self._predict_tree(tree, X, values, np.arange(len(X)))
            # Paper ensembles vote over tree labels; vote fraction supplies a score.
            results.append((values >= .5).astype(float) if self.task == "classification" else values)
        return np.mean(results, axis=0)

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("Regression model has no predict_proba")
        p = self.predict_values(X)
        return np.column_stack([1 - p, p])
