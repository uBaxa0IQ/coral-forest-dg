"""Run the immutable v2 job registry. Set thread limits before scientific imports."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[key]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
import traceback
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning,module=r"erasplit.*")
from nir_dg.series_v2.engine import RUN,freeze,verify_freeze,run_job,write_json,now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.freeze:
        f=freeze();print("FROZEN",len(f["jobs"]),"jobs",flush=True)
        return
    f=verify_freeze()
    todo=[s for s in f["protocol"]["jobs"] if not (RUN/"jobs"/s["job_id"]/"result.json").exists()]
    if args.limit:
        todo=todo[:args.limit]
    todo.sort(key=lambda s:(s["kind"]!="real",s["replica"],s["dataset"],s["scenario"],s["k"],s["n"],s["imbalance"],s["model_seed"]))
    start=time.monotonic();completed=0;failed=[]
    print(f"START {len(todo)} pending jobs, {args.workers} workers",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending={executor.submit(run_job,s):s for s in todo}
        for future in as_completed(pending):
            s=pending[future]
            try:
                r=future.result();completed+=1
                print(json.dumps({"done":completed,"total":len(todo),"elapsed_seconds":round(time.monotonic()-start),**r}),flush=True)
            except Exception:
                error=dict(spec=s,at=now(),traceback=traceback.format_exc())
                failed.append(error)
                write_json(RUN/"failures"/(s["job_id"]+".json"),error)
                print("FAILED",s["job_id"],error["traceback"],flush=True)
            write_json(RUN/"progress.json",dict(updated_at=now(),completed_this_invocation=completed,pending_this_invocation=len(todo)-completed-len(failed),failed_this_invocation=len(failed),elapsed_seconds=time.monotonic()-start,total_registered=len(f["protocol"]["jobs"])))
    write_json(RUN/"execution_summary.json",dict(ended_at=now(),completed_this_invocation=completed,failed=failed,seconds=time.monotonic()-start))
    if failed:
        raise SystemExit(1)


if __name__=="__main__":
    main()
