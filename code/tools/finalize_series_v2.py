"""Wait for the declared inventory, then audit, analyze, export and render.

No training mutation, new selection, network access or message sending.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"runs/expanded_v2"
OUT=ROOT/"reports/series_v2"
BUNDLE=Path("C:/Users/uBaxa/.cache/codex-runtimes/codex-primary-runtime/dependencies")
MARKER=Path("C:/Users/uBaxa/.codex/plugins/cache/openai-primary-runtime/pdf/26.1026.10807/skills/pdf/container_tools/mark_artifact_operation_started.mjs")


def main():
    specs=json.loads((RUN/"frozen_protocol.json").read_text(encoding="utf-8"))["protocol"]["jobs"]
    print("Waiting for",len(specs),"registered jobs",flush=True)
    last=-1
    while True:
        completed=sum((RUN/"jobs"/s["job_id"]/"result.json").exists() for s in specs)
        failures=list((RUN/"failures").glob("*.json")) if (RUN/"failures").exists() else []
        addendum_path=RUN/"source_failure_addendum.json"
        accepted=set(json.loads(addendum_path.read_text(encoding="utf-8"))["addendum"]["jobs"]) if addendum_path.exists() else set()
        if failures and not {p.stem for p in failures}<=accepted:
            print("FAILURES require investigation; automatic finalization stopped",len(failures),flush=True)
            raise SystemExit(1)
        if completed!=last:
            print(datetime.now(timezone.utc).isoformat(),completed,"complete",flush=True);last=completed
        if completed==len(specs):
            break
        time.sleep(30)
    env={**os.environ,**{k:"1" for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]}}
    subprocess.run([sys.executable,"-X","utf8","tools/audit_series_v2.py"],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,"-X","utf8","tools/analyze_series_v2.py"],cwd=ROOT,env=env,check=True)
    subprocess.run([str(BUNDLE/"node/bin/node.exe"),str(MARKER),"--operation-kind","create","--expected-output-count","1","--output-format","pdf"],cwd=ROOT,check=True)
    subprocess.run([str(BUNDLE/"python/python.exe"),"-X","utf8","tools/export_series_v2_pdf.py"],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-X","utf8","tools/verify_series_v2_pdf.py"],cwd=ROOT,env=env,check=True)
    print("FINALIZATION COMPLETE",flush=True)


if __name__=="__main__":
    main()
