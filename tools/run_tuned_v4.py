"""Staged driver: prepare -> source -> fit/seal -> publish -> evaluate -> report."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_versions.tuned_v4.engine import *
from research_versions.tuned_v4.analysis import source_summary,analyze
from research_versions.tuned_v4.checks import replay_source,replay_model


def batch(function,items,stage,workers):
    started=time.monotonic();complete=0;failed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(function,item):item for item in items}
        for future in as_completed(futures):
            item=futures[future]
            try:future.result();complete+=1
            except Exception:failed.append(dict(item=item,error=traceback.format_exc()));write(RUN/f'{stage}_failures.json',failed)
            progress=dict(stage=stage,completed=complete,failed=len(failed),planned=len(items),elapsed_seconds=time.monotonic()-started,updated_at=now())
            write(RUN/'progress.json',progress);print(json.dumps(progress),flush=True)
    if failed:raise RuntimeError(f'{stage} has {len(failed)} failed jobs; no automatic target continuation')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['prepare','source','fit','evaluate','report']);parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    if args.stage=='prepare':print(json.dumps(dict(source_protocol_sha256=prepare()['sha256'])));return
    prepare()
    if args.stage=='source':
        items=[(c,m) for c in contexts() for m in methods(c['task'])]
        batch(search_family,items,'source',args.workers)
        batch(replay_source,items,'source_replay',args.workers)
        print(json.dumps(source_summary()));return
    if args.stage=='fit':
        if len(list((RUN/'selection').glob('*.json')))!=127:raise RuntimeError('All source selections required')
        batch(source_fit,final_jobs(),'fit',args.workers);print(json.dumps(dict(final_protocol_sha256=seal()['sha256'])));return
    if args.stage=='evaluate':
        frozen=data.authorization();assert hashes()==frozen['protocol']['runtime_hashes']
        # The checked-in final protocol must exist in the public remote tracking commit.
        commit=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
        blob=subprocess.check_output(['git','show',commit+':runs/tuned_v4/final_protocol.json'],cwd=ROOT)
        assert sha256(blob).digest()==sha256((RUN/'final_protocol.json').read_bytes()).digest(),'Publish frozen final protocol before target access'
        for p,h in {**frozen['protocol']['selection_sha256'],**frozen['protocol']['source_receipt_sha256']}.items():assert sha256((ROOT/p).read_bytes()).hexdigest()==h,p
        access=RUN/'global_target_access.json'
        if not access.exists():write(access,dict(at=now(),published_commit=commit,final_protocol_sha256=frozen['sha256'],new_acs_states=NEW_STATES,synthetic_namespace='sealed_v4_target',policy='no changes or automatic retries after target access'))
        # Serial preparation prevents concurrent downloads/cache writers.
        data.acs_target('acs_income');data.acs_target('acs_public_coverage')
        batch(evaluate,final_jobs(),'evaluate',args.workers)
        batch(replay_model,final_jobs(),'model_replay',args.workers)
        print(json.dumps(analyze())[:1000]);return
    if args.stage=='report':print(json.dumps(analyze())[:1000])


if __name__=='__main__':main()
