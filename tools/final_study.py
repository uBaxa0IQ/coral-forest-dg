import argparse
from pathlib import Path
from nir_dg.datasets import ROOT
from nir_dg.target_evaluation import evaluate_registered, preregister

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["freeze", "evaluate"])
    p.add_argument("--run-dir", default="runs/final_v1")
    p.add_argument("--datasets", nargs="*")
    args = p.parse_args()
    directory = ROOT / args.run_dir
    if args.command == "freeze":
        names = ["acs_income", "hospital_readmission", "beijing_pm25", "gas_turbine_co"]
        paths = [ROOT / f"runs/study_cv_v1_{name}/source_cv_metrics.csv" for name in names]
        value = preregister(paths, directory)
        print(value["protocol_hash"], len(value["design"]["experiment_manifest"]))
    else:
        evaluate_registered(directory, args.datasets)
