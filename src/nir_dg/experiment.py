from __future__ import annotations
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import traceback

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import yaml

from nir_dg.datasets import ROOT, load_source
from nir_dg.metrics import score
from nir_dg.models import Adapter, METHODS, applicable_methods
from nir_dg.preprocessing import make_preprocessor
from nir_dg.protocol import ProtocolError, code_fingerprint, digest_json, source_folds


def append_record(path, record):
    path = Path(path)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def system_info():
    packages = {}
    for name in ["numpy", "scipy", "pandas", "scikit-learn", "erasplit", "anchorboosting", "lightgbm", "cvxpy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return dict(platform=platform.platform(), python=platform.python_version(), logical_cpu_count=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total, packages=packages, git_commit=commit)


def run_source_cv(config_path, run_dir, datasets=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("target_access") is not False:
        raise ProtocolError("Source CV config must explicitly disable target access")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = run_dir / "provenance.json"
    config_hash, code_hash = digest_json(config), code_fingerprint(ROOT)
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf-8"))
        if old["config_hash"] != config_hash or old["code_hash"] != code_hash:
            raise ProtocolError("Run code/config changed: choose a new run directory")
    else:
        provenance_path.write_text(json.dumps(dict(config_hash=config_hash, code_hash=code_hash, started_at=datetime.now(timezone.utc).isoformat(), target_access=False), indent=2), encoding="utf-8")
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        (run_dir / "system_info.json").write_text(json.dumps(system_info(), indent=2), encoding="utf-8")
    results_path = run_dir / "source_cv_metrics.csv"
    done = set()
    if results_path.exists():
        done = set(pd.read_csv(results_path).experiment_id)
    datasets = datasets or config["datasets"]
    for name in datasets:
        data = load_source(name)
        meta = dict(source_fingerprint=data.fingerprint(), source_rows=len(data.y), source_domains=np.unique(data.environments).tolist(), target_access=False)
        meta_path = run_dir / f"{name}_source.json"
        if meta_path.exists() and json.loads(meta_path.read_text())["source_fingerprint"] != meta["source_fingerprint"]:
            raise ProtocolError("Source data changed during a run")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for fold, train, valid in source_folds(data, config["max_folds"]):
            preprocessor = make_preprocessor(data.X.iloc[train])
            Xt = preprocessor.fit_transform(data.X.iloc[train])
            Xv = preprocessor.transform(data.X.iloc[valid])
            for seed in config["seeds"]:
                for method in applicable_methods(data.task):
                    for trial, params in enumerate(config["hpo"][method]):
                        experiment_id = f"{name}:{fold}:{seed}:{method}:{trial}"
                        if experiment_id in done:
                            continue
                        common = dict(experiment_id=experiment_id, dataset=name, task=data.task, fold=fold, seed=seed, method=method, trial=trial, parameters=json.dumps(params, sort_keys=True), implementation_status=METHODS[method].fidelity, n_train=len(train), n_valid=len(valid), n_features=Xt.shape[1])
                        started = perf_counter()
                        try:
                            with threadpool_limits(limits=1):
                                model = Adapter(method, data.task, seed, params).fit(Xt, data.y[train], data.environments[train])
                                fit_seconds = perf_counter() - started
                                prediction_started = perf_counter()
                                prediction = model.predict_values(Xv)
                                predict_seconds = perf_counter() - prediction_started
                            metrics = score(data.task, data.y[valid], prediction)
                            row = dict(common, outcome="ok", primary=metrics["roc_auc" if data.task == "classification" else "rmse"], metrics=json.dumps(metrics, sort_keys=True), fit_seconds=fit_seconds, predict_seconds=predict_seconds, rss_mb=psutil.Process().memory_info().rss / 2**20, error="")
                        except Exception as error:
                            row = dict(common, outcome="failed", primary=np.nan, metrics="{}", fit_seconds=perf_counter() - started, predict_seconds=np.nan, rss_mb=psutil.Process().memory_info().rss / 2**20, error=repr(error))
                            with (run_dir / "errors.log").open("a", encoding="utf-8") as f:
                                f.write(experiment_id + "\n" + traceback.format_exc() + "\n")
                        append_record(results_path, row)
                        done.add(experiment_id)
                        print(experiment_id, row["outcome"], f"{row['fit_seconds']:.2f}s", flush=True)
    return results_path
