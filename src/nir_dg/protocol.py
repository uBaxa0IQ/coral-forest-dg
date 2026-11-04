from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


class ProtocolError(RuntimeError):
    pass


def digest_json(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SourceData:
    X: pd.DataFrame
    y: np.ndarray
    environments: np.ndarray
    row_ids: np.ndarray
    task: str
    name: str
    temporal: bool = False
    time_ids: np.ndarray | None = None
    entity_ids: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.X)
        if any(len(a) != n for a in (self.y, self.environments, self.row_ids)):
            raise ProtocolError("Data arrays have different lengths")
        if len(np.unique(self.row_ids)) != n:
            raise ProtocolError("Duplicate row identifiers")
        if self.task not in {"classification", "regression"}:
            raise ProtocolError("Unsupported task")
        if not n or pd.isna(self.y).any():
            raise ProtocolError("Empty source or missing outcomes")
        if self.temporal and self.time_ids is None:
            raise ProtocolError("Temporal data require explicit time identifiers")
        if self.time_ids is not None and len(self.time_ids) != n:
            raise ProtocolError("Time identifiers have incorrect length")
        if self.entity_ids is not None and len(self.entity_ids) != n:
            raise ProtocolError("Entity identifiers have incorrect length")

    def fingerprint(self):
        h = sha256()
        h.update(pd.util.hash_pandas_object(self.X, index=False).values.tobytes())
        for values in (self.y, self.environments, self.row_ids):
            h.update(pd.util.hash_array(np.asarray(values)).tobytes())
        return h.hexdigest()


def source_folds(data: SourceData, max_folds=None):
    envs = np.unique(data.environments)
    if len(envs) < 3:
        raise ProtocolError("At least three source environments required for domain CV")
    if data.temporal:
        envs = sorted(envs, key=lambda e: np.min(data.time_ids[data.environments == e]))
        chosen = list(range(2, len(envs)))
    else:
        chosen = list(range(len(envs)))
    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be positive")
        # Source IDs/order alone select folds, before any model scores exist.
        chosen = chosen[-max_folds:]
    for j in chosen:
        held_out = envs[j]
        valid = np.flatnonzero(data.environments == held_out)
        permitted = envs[:j] if data.temporal else [e for e in envs if e != held_out]
        train = np.flatnonzero(np.isin(data.environments, permitted))
        if data.entity_ids is not None:
            train = train[~np.isin(data.entity_ids[train], data.entity_ids[valid])]
        if len(np.unique(data.environments[train])) < 2:
            raise ProtocolError("Entity exclusion leaves fewer than two training domains")
        if data.temporal and np.max(data.time_ids[train]) >= np.min(data.time_ids[valid]):
            raise ProtocolError("Training time must strictly precede validation time")
        if set(data.row_ids[train]) & set(data.row_ids[valid]):
            raise ProtocolError("Training and validation rows overlap")
        yield str(held_out), train, valid


def code_fingerprint(root):
    root = Path(root)
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "tools").glob("*.py"))
    return digest_json({p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths})


def freeze_protocol(run_dir, design, root):
    """Freeze the design. This never reads any target features, labels or summaries."""
    path = Path(run_dir) / "frozen_protocol.json"
    if path.exists():
        raise ProtocolError("Protocol already frozen; create a separately documented run")
    required = {"datasets", "methods", "seeds", "metrics", "hpo", "source_folds", "target_ids", "experiment_manifest", "source_fingerprints"}
    if missing := required - design.keys():
        raise ProtocolError(f"Incomplete preregistration: {sorted(missing)}")
    if not all(design[k] for k in required):
        raise ProtocolError("Preregistration fields may not be empty")
    value = dict(design=design, code_hash=code_fingerprint(root), frozen_at=datetime.now(timezone.utc).isoformat())
    value["protocol_hash"] = digest_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    return value


def verified_protocol(run_dir, root):
    path = Path(run_dir) / "frozen_protocol.json"
    if not path.exists():
        raise ProtocolError("Target is locked until protocol freeze")
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = value.pop("protocol_hash")
    if digest_json(value) != recorded:
        raise ProtocolError("Frozen protocol has been modified")
    if code_fingerprint(root) != value["code_hash"]:
        raise ProtocolError("Source code changed after freeze")
    return dict(value, protocol_hash=recorded)


class TargetVault:
    """No target loader is called before integrity and experiment identity checks."""
    def __init__(self, run_dir, root, loader):
        self.run_dir, self.root, self._loader = Path(run_dir), Path(root), loader

    def load_for_evaluation(self, experiment_id):
        protocol = verified_protocol(self.run_dir, self.root)
        entries = protocol["design"]["experiment_manifest"]
        if experiment_id not in entries:
            raise ProtocolError("Experiment was not preregistered")
        receipts = self.run_dir / "target_access"
        receipts.mkdir(exist_ok=True)
        # Exclusive receipt persists even if evaluation fails: no silent fresh holdout.
        receipt = receipts / (sha256(experiment_id.encode()).hexdigest() + ".json")
        try:
            with receipt.open("x", encoding="utf-8") as f:
                json.dump(dict(experiment_id=experiment_id, protocol_hash=protocol["protocol_hash"], accessed_at=datetime.now(timezone.utc).isoformat()), f)
        except FileExistsError as error:
            raise ProtocolError("Target already accessed for this experiment; document a deviation") from error
        return self._loader(entries[experiment_id])
