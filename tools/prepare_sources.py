from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nir_dg.datasets import prepare_acs, prepare_beijing, prepare_hospital, prepare_synthetic

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = {"acs_income": prepare_acs, "hospital_readmission": prepare_hospital, "beijing_pm25": prepare_beijing,
            "synthetic_classification": lambda: prepare_synthetic("classification"), "synthetic_regression": lambda: prepare_synthetic("regression")}
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(f): name for name, f in jobs.items()}
        for job in as_completed(pending):
            name = pending[job]
            try:
                results[name] = dict(status="source_prepared", **job.result())
            except Exception as error:
                results[name] = dict(status="blocked", error=repr(error))
            print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)
    path = Path(__file__).resolve().parents[1] / "reports/source_preparation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
