"""Create local compatibility views without changing frozen experiment bytes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE=Path(__file__).resolve().parent
WORKSPACE=CODE.parent
VIEWS={
    'docs':WORKSPACE/'materials/docs',
    'literature':WORKSPACE/'materials/literature',
    'research_input':WORKSPACE/'materials/research_input',
    'reports':WORKSPACE/'nir/results',
}


def setup():
    for name,target in VIEWS.items():
        link=CODE/name
        if not target.is_dir():raise RuntimeError(f'Missing canonical directory: {target}')
        if link.exists():
            if link.resolve()!=target.resolve():raise RuntimeError(f'Refusing to replace unrelated directory: {link}')
            continue
        if os.name=='nt':
            quote=lambda s:"'"+str(s).replace("'","''")+"'"
            cmd='New-Item -ItemType Junction -Path '+quote(link)+' -Value '+quote(target)+' | Out-Null'
            subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',cmd],check=True)
        else:link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)
    # Editable .pth only belongs to the local environment, outside frozen code.
    for p in (CODE/'.venv/Lib/site-packages').glob('__editable__.coral_forest_dg-*.pth'):
        p.write_text(str(CODE/'src')+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'workspace':str(WORKSPACE),'compatibility_views':{k:str(v) for k,v in VIEWS.items()}},ensure_ascii=False))


def verify():
    sys.path[:0]=[str(CODE),str(CODE/'src')]
    from hashlib import sha256
    import importlib.metadata
    from nir_dg.series_v2.engine import verify_freeze
    from research_versions.tuned_v4.engine import hashes
    v2=verify_freeze()
    for protocol in ['source_protocol.json','final_protocol.json']:
        frozen=json.loads((CODE/'runs/tuned_v4'/protocol).read_text(encoding='utf-8'))
        assert hashes()==frozen['protocol']['runtime_hashes'],protocol
        for package,version in frozen['protocol'].get('packages',{}).items():
            assert importlib.metadata.version(package)==version,package
    source=json.loads((CODE/'runs/expanded_v2/frozen_protocol.json').read_text(encoding='utf-8'))
    for name,h in source['protocol'].get('research_documents_sha256',{}).items():
        assert sha256((CODE/name).read_bytes()).hexdigest()==h,name
    result={'v2_protocol_sha256':v2['sha256'],'v2_runtime_packages_sources_verified':True,'v2_frozen_document_hashes_verified':True,'v4_source_and_final_runtime_hashes_verified':True,'v4_recorded_package_versions_verified':True,'new_experiments_run':False}
    print(json.dumps(result,ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['setup','verify']);args=parser.parse_args()
    setup()
    if args.command=='verify':verify()


if __name__=='__main__':main()
