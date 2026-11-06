"""Check retained handoff integrity and completed source-only gates before P2."""
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from nir_dg.datasets import ROOT
from nir_dg.target_evaluation import select_source_configs, TARGET_IDS

if __name__ == "__main__":
    package = ROOT / "research_input/NIR_Codex_Handoff_v2"
    manifest = json.loads((package / "FILE_HASHES.json").read_text(encoding="utf-8"))
    for entry in manifest["items"]:
        assert sha256((package / entry["file"]).read_bytes()).hexdigest() == entry["sha256"], entry["file"]
    paths = [ROOT / f"runs/study_cv_v1_{name}/source_cv_metrics.csv" for name in TARGET_IDS]
    data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert len(data) == 1050, len(data)
    assert data.outcome.eq("ok").all()
    choices = select_source_configs(paths, list(TARGET_IDS))
    assert len(choices) == 126
    downloaded = {e["id"]: e for e in json.loads((ROOT / "literature/download_manifest.json").read_text(encoding="utf-8"))}
    assert all(downloaded[i]["status"] == "downloaded" for i in ["irf", "crf", "meta_forests", "maxrm_rf", "era_splitting", "anchor_boosting"])
    test = ET.parse(ROOT / "reports/tests.xml").getroot()
    suites = [test] if test.tag == "testsuite" else list(test)
    assert sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites) == 0
    result = dict(handoff_hashes_verified=len(manifest["items"]), source_fits=len(data), successful_source_fits=int(data.outcome.eq("ok").sum()), preregisterable_model_seed_runs=len(choices), core_method_pdfs=6, unit_tests=sum(int(s.attrib.get("tests", 0)) for s in suites), target_metrics_exist=(ROOT / "runs/final_v1/target_metrics.csv").exists())
    (ROOT / "reports/preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
