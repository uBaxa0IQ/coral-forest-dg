"""Unattended sequential study; no model calls while computations run.

Stops on errors. Publishes source/final protocols before their protected stages.
Does not relax failed solvers or silently change any candidate or target.
"""
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/tuned_v4';OUT=ROOT/'reports/tuned_v4'
PYTHON=ROOT/'.venv/Scripts/python.exe'
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n');tmp.replace(path)


def stamp():return datetime.now(timezone.utc).isoformat()


def command(args,log):
    with (OUT/log).open('a',encoding='utf-8') as output:
        subprocess.run(args,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,creationflags=FLAGS,check=True)


def publish(message,paths):
    command(['git','add','--',*paths],'publication.log')
    delta=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT,creationflags=FLAGS)
    if delta.returncode==1:command(['git','commit','-m',message],'publication.log')
    elif delta.returncode!=0:raise RuntimeError('Cannot inspect Git index')
    command(['git','push','origin','main'],'publication.log')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def main():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    lock=RUN/'coordinator.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    state=dict(started_at=stamp(),pid=os.getpid(),status='running',history=[])
    prereg=ROOT/'docs/tuned_v4/preregistration.md';prereg_sha=sha256(prereg.read_bytes()).hexdigest()
    def stage(name):
        assert sha256(prereg.read_bytes()).hexdigest()==prereg_sha,'Preregistration changed during study'
        state.update(stage=name,updated_at=stamp());state['history'].append(dict(stage=name,started_at=stamp()));write(RUN/'coordinator.json',state)
        command([str(PYTHON),'tools/run_tuned_v4.py','--stage',name,'--workers','6'],name+'.log')
        state['history'][-1]['completed_at']=stamp();write(RUN/'coordinator.json',state)
    try:
        stage('source')
        state['source_results_commit']=publish('Record source-only HPO selections and deterministic replays v4',['runs/tuned_v4','reports/tuned_v4'])
        stage('fit')
        write(RUN/'research_document_freeze.json',dict(preregistration_sha256=prereg_sha,source_protocol_sha256=sha256((RUN/'source_protocol.json').read_bytes()).hexdigest(),coordinator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),frozen_at=stamp()))
        state['final_protocol_commit']=publish('Freeze tuned v4 source models and fresh target protocol',['runs/tuned_v4','reports/tuned_v4'])
        stage('evaluate')
        command([str(PYTHON),'tools/finish_tuned_v4_report.py'],'report.log')
        state.update(status='complete',completed_at=stamp());write(RUN/'coordinator.json',state)
        state['results_commit']=publish('Publish fresh target results, statistics and report v4',['runs/tuned_v4','reports/tuned_v4','configs/target_exposure_ledger_v4.json'])
        # Final publication receipt remains local to avoid a self-referential commit.
        write(OUT/'publication_receipt.json',state)
    except Exception:
        state.update(status='stopped_on_error',error=traceback.format_exc(),updated_at=stamp());write(RUN/'coordinator.json',state)
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
