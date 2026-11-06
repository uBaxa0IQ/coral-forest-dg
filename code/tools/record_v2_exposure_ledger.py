"""Persist the consumed-holdout boundary for subsequent research versions."""
import json
from pathlib import Path
from nir_dg.series_v2.engine import ROOT,RUN,verify_freeze,write_json,now


if __name__=="__main__":
    f=verify_freeze();receipts={}
    for s in f["protocol"]["jobs"]:
        p=RUN/"jobs"/s["job_id"] /"target_access.json"
        if p.exists():
            r=json.loads(p.read_text(encoding="utf-8"));key=s["dataset"] if s["kind"]=="real" else f"synthetic:{s['task']}:{s['scenario']}"
            receipts[key]=min(receipts.get(key,r["authorized_at"]),r["authorized_at"])
    result=dict(recorded_at=now(),protocol_sha256=f["sha256"],consumed_real_targets={"acs_2018":{"states":["MN","OH","PA","OR","VA"],"joint_task_family":["ACSIncome","ACSPublicCoverage"]},"beijing_station":["Nongzhanguan","Tiantan","Wanliu","Wanshouxigong"],"appliances_time":["2016-04-02/2016-04-30","2016-05-01/2016-05-27"]},consumed_synthetic_draws={"root_seed":2026111401,"target_namespace":"sealed_target","replicas":list(range(10)),"tasks":["classification","regression"],"scenarios":"registered generators; no regression label_prior","shared_across":"K/N/imbalance/model-seed within task/scenario/data-replica"},earliest_authorization_receipts=receipts,evidence_limit="Framework receipt times, not a trusted OS access log",policy="These targets and historically exposed v1 domains are no longer independent final tests for new decisions. Do not relabel another seed/resample as a fresh real target. Prospective new synthetic draws are new MC data within the same DGP, not evidence for a new real domain family.")
    write_json(ROOT/"configs/target_exposure_ledger_v2.json",result)
    print("Consumed target ledger recorded")
