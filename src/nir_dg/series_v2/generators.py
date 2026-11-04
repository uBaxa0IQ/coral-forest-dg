"""Explicit tabular DGPs, with independent source/target RNG namespaces."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd

SCENARIOS = ["iid", "covariate", "spurious", "concept", "heteroscedastic", "missingness", "label_prior"]
ROOT_SEED = 2026111401


def seed_for(*parts):
    payload = json.dumps([ROOT_SEED, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(sha256(payload.encode()).digest()[:8], "little") % (2**32-1)


def allocation(n, k, ratio=1., replica=0):
    weights = np.geomspace(1., float(ratio), k)
    weights = weights[np.random.default_rng(seed_for("size_order", replica, k)).permutation(k)]
    real = n * weights / weights.sum()
    counts = np.floor(real).astype(int)
    counts[np.argsort(-(real-counts), kind="stable")[:n-counts.sum()]] += 1
    assert counts.sum() == n and counts.min() > 0
    return counts


def draw(task, scenario, n, domain_parameter, rng):
    """Eight features; a single (n,12) draw makes prefixes nested across N."""
    if scenario == "label_prior" and task != "classification":
        raise ValueError("Pure class-prior shift has no registered regression variant")
    z = rng.normal(size=(n, 12))
    X = z[:, :8].copy()
    delta = float(domain_parameter)
    if scenario == "covariate":
        X[:, 0] += delta
        X[:, 1] *= np.exp(.25 * delta)
    noise_scale = np.exp(.55 * delta) if scenario == "heteroscedastic" else .7
    beta = 1.5 + .8 * delta if scenario == "concept" else 1.5
    signal = beta * X[:, 0] + .5 * np.sin(X[:, 1])
    latent = signal + noise_scale * z[:, 8]
    y = (latent > 0).astype(int) if task == "classification" else latent
    if scenario == "spurious":
        # Source a=.75..2.25; targets reverse that anticausal association.
        X[:, 2] = delta * (2*y-1 if task == "classification" else y) + z[:, 9]
    elif scenario == "label_prior":
        from scipy.special import ndtr
        y = (ndtr(z[:, 10]) < delta).astype(int)
        X[:, 0] = .8 * (2*y-1) + z[:, 0]
        X[:, 1] = .4 * (2*y-1) + z[:, 1]
    elif scenario == "missingness":
        # Missingness is a measurement shift, not claimed pure covariate shift.
        from scipy.special import ndtr
        mask = ndtr(z[:, 9:12]) < delta
        X[:, :3][mask] = np.nan
    return X, y


def parameters(scenario, k, role):
    if role == "source":
        if scenario == "spurious":
            return np.linspace(.75, 2.25, k)
        if scenario == "label_prior":
            return np.linspace(.2, .8, k)
        if scenario == "missingness":
            return np.full(k, .05)
        if scenario == "iid":
            return np.zeros(k)
        return np.linspace(-1., 1., k)
    return {"iid": [0., 0., 0.], "covariate": [-2., 0., 2.], "spurious": [-.75, -1.5, -2.25], "concept": [-2.5, 0., 2.5], "heteroscedastic": [-1.5, 0., 1.5], "missingness": [.15, .3, .5], "label_prior": [.1, .5, .9]}[scenario]


def source_frame(spec):
    frames=[]
    counts=allocation(spec["n"],spec["k"],spec["imbalance"],spec["replica"])
    for i,(count,parameter) in enumerate(zip(counts,parameters(spec["scenario"],spec["k"],"source"))):
        # Shared mechanism parameters retain common random draws across K.
        # IID/missingness domains share a parameter but must remain distinct draws.
        stream_key=i if spec["scenario"] in {"iid","missingness"} else round(float(parameter),12)
        rng=np.random.default_rng(seed_for("source",spec["task"],spec["scenario"],spec["replica"],stream_key))
        X,y=draw(spec["task"],spec["scenario"],int(count),parameter,rng)
        f=pd.DataFrame(X,columns=[f"x{j}" for j in range(X.shape[1])])
        f["__y"],f["__env"]=y,f"source_{i:02d}"
        f["__row_id"]=[f"s:{i}:{j}" for j in range(len(f))]
        frames.append(f)
    return pd.concat(frames,ignore_index=True)


def target_frame(spec, n_per_domain=2000, authorization=None):
    """Caller must verify frozen protocol; source N/K/imbalance/model seed excluded.

    This deliberately shares the SAME holdout draws across planned factorial
    contrasts in a data replica. These are paired comparisons, not new datasets.
    """
    from .engine import verify_freeze
    if authorization is None or verify_freeze()["sha256"] != authorization:
        raise RuntimeError("New synthetic target requires frozen protocol authorization")
    frames=[]
    for i,parameter in enumerate(parameters(spec["scenario"],spec["k"],"target")):
        rng=np.random.default_rng(seed_for("sealed_target",spec["task"],spec["scenario"],spec["replica"],i))
        X,y=draw(spec["task"],spec["scenario"],n_per_domain,parameter,rng)
        f=pd.DataFrame(X,columns=[f"x{j}" for j in range(X.shape[1])])
        f["__y"],f["__env"]=y,f"new_target_{i:02d}"
        f["__row_id"]=[f"t:{spec['scenario']}:{spec['replica']}:{i}:{j}" for j in range(len(f))]
        frames.append(f)
    return pd.concat(frames,ignore_index=True)
