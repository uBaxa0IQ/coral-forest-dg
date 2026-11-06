"""Verify exact relocated bytes in Git's index, including raw CRLF receipts."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'materials/repository_layout'


def main():
    before=json.loads((OUT/'before.json').read_text(encoding='utf-8'))
    entries=[r for r in before['files'] if r['old'] not in {'.gitignore','.gitattributes'}]
    original_count=len(entries)
    template_paths=['materials/templates/coursework/preambula.original.tex','materials/templates/coursework/report.original.tex','materials/templates/nir_structure/report.original.tex','nir/report/preambula.tex']
    entries += [dict(new=p,sha256=sha256((ROOT/p).read_bytes()).hexdigest()) for p in template_paths]
    response=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=('\n'.join(':'+r['new'] for r in entries)+'\n').encode(),stdout=subprocess.PIPE,check=True).stdout
    pos=0;bad=[]
    for entry in entries:
        end=response.index(b'\n',pos);header=response[pos:end].split();pos=end+1
        if header[-1]==b'missing':bad.append(dict(path=entry['new'],reason='not staged'));continue
        size=int(header[-1]);blob=response[pos:pos+size];pos+=size+1
        if sha256(blob).hexdigest()!=entry['sha256']:bad.append(dict(path=entry['new'],reason='staged bytes changed'))
    tracked=subprocess.check_output(['git','ls-files','--cached','-z'],cwd=ROOT).decode().split('\0')
    unwanted=[p for p in tracked if p and (p.startswith('code/.venv/') or p.startswith('code/data/') and p!='code/data/README.md' or p.startswith('materials/literature/pdfs/') and not p.endswith('.gitkeep') or p.startswith('code/runs/') and ('/models/' in p or '/predictions/' in p))]
    value=dict(original_artifact_blobs_verified=original_count,template_source_blobs_verified=len(template_paths),staged_byte_mismatches=bad,private_data_or_environment_files_staged=unwanted,scientific_git_bytes_preserved=not bad)
    (OUT/'git_byte_preservation.json').write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(value));assert not bad and not unwanted


if __name__=='__main__':main()
