import argparse
from pathlib import Path
from nir_dg.datasets import ROOT


def main():
    parser = argparse.ArgumentParser(description="Source-only tree DG research bench")
    sub = parser.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("source-cv", help="Run source-only validation; no target loading")
    cv.add_argument("--config", default=str(ROOT / "configs/source_pilot.yaml"))
    cv.add_argument("--run-dir", required=True)
    cv.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    if args.command == "source-cv":
        from nir_dg.experiment import run_source_cv
        run_source_cv(args.config, args.run_dir, args.datasets)


if __name__ == "__main__":
    main()
