"""Check staged scientific artifacts in one git cat-file batch."""
import hashlib
import json
from pathlib import Path
import subprocess

paths=subprocess.check_output(["git","diff","--cached","--name-only"],text=True).splitlines()
selected=[p for p in paths if p.startswith(("runs/expanded_v2/","reports/series_v2/"))]
request=("\n".join(":"+p for p in selected)+"\n").encode()
result=subprocess.run(["git","cat-file","--batch"],input=request,stdout=subprocess.PIPE,check=True).stdout
position=0;bad=[];total=0
for path in selected:
    end=result.index(b"\n",position);header=result[position:end].split();size=int(header[2]);position=end+1
    blob=result[position:position+size];position+=size+1;total+=size
    if hashlib.sha256(blob).digest()!=hashlib.sha256(Path(path).read_bytes()).digest():
        bad.append(path)
report=dict(staged_artifacts=len(selected),checked_bytes=total,mismatches=bad,git_line_ending_preservation_verified=not bad)
Path("reports/series_v2/git_byte_preservation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report));assert not bad
