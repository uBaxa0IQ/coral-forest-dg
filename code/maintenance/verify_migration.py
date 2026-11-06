"""One byte-level pass over relocated tracked files; no model retraining."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'materials/repository_layout'


def main():
    before=json.loads((OUT/'before.json').read_text(encoding='utf-8'))
    deliberate={'.gitignore','.gitattributes'};missing=[];changed=[];checked=0;total=0
    for entry in before['files']:
        if entry['old'] in deliberate:continue
        p=ROOT/entry['new']
        if not p.is_file():missing.append(entry);continue
        value=p.read_bytes();total+=len(value);checked+=1
        if sha256(value).hexdigest()!=entry['sha256']:changed.append(entry)
    report=dict(verified_at=datetime.now(timezone.utc).isoformat(),pre_migration_commit=before['commit'],checked_files=checked,checked_bytes=total,expected_modified_root_metadata=sorted(deliberate),missing=missing,changed=changed,scientific_content_bytes_preserved=not missing and not changed,new_experiments_run=False)
    (OUT/'byte_preservation.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(report));assert not missing and not changed


if __name__=='__main__':main()
