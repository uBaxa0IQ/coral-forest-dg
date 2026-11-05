"""Explicit salvage of predeclared non-MaxRM pipelines after source solver failures.

No solver relaxation, new grid, target-based decision or training-runtime edit.
The omission and this driver are committed before these jobs access target.
"""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[k]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from hashlib import sha256
import json
from pathlib import Path
from nir_dg.series_v2 import engine

MANIFEST=engine.RUN/"source_failure_addendum.json"


def register():
    frozen=engine.verify_freeze()
    failures=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((engine.RUN/"failures").glob("*.json"))]
    assert len(failures)==5
    for f in failures:
        assert "Unverified MaxRM optimum: optimal_inaccurate" in f["traceback"]
        assert not (engine.RUN/"jobs"/f["spec"]["job_id"] /"target_access.json").exists()
    body=dict(recorded_at=engine.now(),original_protocol_sha256=frozen["sha256"],driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),jobs=[f["spec"]["job_id"] for f in failures],omitted_method="maxrm",reason="Source-stage CLARABEL optimal_inaccurate rejected by frozen implementation",policy="Preserve original failures. Salvage only other originally registered method/config pipelines, unchanged data, code, seeds and selection. Both MaxRM grid points omitted; no baseline fallback. Previously opened draws are used only to finish ex-ante registered evaluations, not to validate new decisions.",statistics="Require both model seeds for each replica-method cell. Missing MaxRM confirmatory contrast is unperformed; conservative Holm input p=1 keeps original family size nine. Complete-case estimates secondary only.")
    engine.write_json(MANIFEST,{"addendum":body,"sha256":engine.digest(body)})
    print(json.dumps(body,indent=2))


def recover(spec):
    a=json.loads(MANIFEST.read_text(encoding="utf-8"));body=a["addendum"]
    assert engine.digest(body)==a["sha256"]
    assert sha256(Path(__file__).read_bytes()).hexdigest()==body["driver_sha256"]
    assert spec["job_id"] in body["jobs"]
    path=engine.RUN/"jobs"/spec["job_id"]
    if (path/"result.json").exists():
        return dict(job_id=spec["job_id"],status="already_completed")
    original_grids=engine.grids
    engine.grids=lambda task:{m:g for m,g in original_grids(task).items() if m!="maxrm"}
    original_source_stage=engine.source_stage
    def source_stage(*args,**kwargs):
        selection,models=original_source_stage(*args,**kwargs)
        selection.update(omitted_methods=["maxrm"],execution_addendum_sha256=a["sha256"],recovery_driver_sha256=body["driver_sha256"])
        engine.write_json(path/"source_selection.json",selection)
        return selection,models
    engine.source_stage=source_stage
    result=engine.run_job(spec)
    record=json.loads((path/"result.json").read_text(encoding="utf-8"))
    record.update(status="complete_with_method_failure",omitted_methods=["maxrm"],execution_addendum_sha256=a["sha256"])
    engine.write_json(path/"result.json",record)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--register",action="store_true");args=parser.parse_args()
    if args.register:
        register()
    else:
        frozen=engine.verify_freeze();a=json.loads(MANIFEST.read_text(encoding="utf-8"))
        specs=[s for s in frozen["protocol"]["jobs"] if s["job_id"] in a["addendum"]["jobs"]]
        with ProcessPoolExecutor(max_workers=5) as pool:
            for result in pool.map(recover,specs):
                print(json.dumps(result),flush=True)
