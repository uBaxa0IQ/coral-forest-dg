"""Bounded verification of new derived material and prospective source replay."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from hashlib import sha256
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from nir_dg.series_v2.engine import verify_freeze,folds
from nir_dg.series_v2.generators import source_frame
from nir_dg.series_v2.data import features
from research_versions.robustness_v3.preprocessing import make_preprocessor
from research_versions.robustness_v3.tuning import ConfiguredModel

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/closeout'


def main():
    frozen=verify_freeze()
    manifest=json.loads((OUT/'evidence_manifest.json').read_text(encoding='utf-8'))
    for p,h in {**manifest['source_artifacts_sha256'],**manifest['derived_files_sha256']}.items():
        assert sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    registry=json.loads((OUT/'claims_registry.json').read_text(encoding='utf-8'))['claims']
    assert len({c['id'] for c in registry})==len(registry)
    for c in registry:
        if 'artifact' in c:assert sha256((ROOT/c['artifact']).read_bytes()).hexdigest()==c['sha256'],c['id']
    a=pd.read_csv(OUT/'tables/primary_tests.csv');b=pd.read_csv(ROOT/'reports/series_v2/tables/hypothesis_tests.csv')
    pd.testing.assert_frame_equal(a.drop(columns='evidence_id'),b)
    protocol=json.loads((OUT/'source_pilot/source_protocol.json').read_text(encoding='utf-8'))
    assert sha256(json.dumps(protocol['protocol'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==protocol['sha256']
    for p,h in protocol['protocol']['code_hashes'].items():assert sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    selected=pd.read_csv(OUT/'source_pilot/selected_source_configurations.csv');replays=[]
    for _,r in selected.iterrows():
        context=next(c for c in protocol['protocol']['contexts'] if c['spec']['task']==r.task);spec=context['spec'];f=source_frame(spec);tr,va=next(folds(f,spec))
        processor=make_preprocessor(f[features(f)].iloc[tr]);X=processor.fit_transform(f[features(f)].iloc[tr]);V=processor.transform(f[features(f)].iloc[va])
        model=ConfiguredModel(r.method,r.task,11,dict(profile=int(r.profile),**json.loads(r.parameters))).fit(X,f.__y.to_numpy()[tr],f.__env.to_numpy()[tr])
        digest=sha256(model.predict(V).tobytes()).hexdigest()
        saved=json.loads((OUT/f'source_pilot/jobs/{r.task}_{int(r.config_id):02d}.json').read_text(encoding='utf-8'))
        assert digest==saved['folds'][0]['prediction_sha256'],(r.task,r.method)
        replays.append(dict(task=r.task,method=r.method,config=int(r.config_id),fold=0,prediction_bytes_identical=True))
    result=dict(frozen_v2_runtime_and_packages_and_source_pools_unchanged=True,protocol_sha256=frozen['sha256'],evidence_hashes_verified=len(manifest['source_artifacts_sha256']),derived_table_hashes_verified=len(manifest['derived_files_sha256']),unique_claim_ids=len(registry),primary_tests_exact_copy=True,pilot_manifest_and_code_hashes_verified=True,source_replays=replays,new_target_evaluations=0)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='passed',source_replays=len(replays),claims=len(registry))))


if __name__=='__main__':main()
