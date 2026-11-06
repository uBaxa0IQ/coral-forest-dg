"""Checksums and first-page identity for newly recovered original PDFs."""
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT/"literature"
NEW={"appliances_energy_paper":"Data driven prediction models", "handoff_06":"Spectrally Deconfounded Random Forests","handoff_17":"Decorr", "handoff_22":"On the Need for a Language", "handoff_31":"TabFSBench"}


if __name__=="__main__":
    records=json.loads((LIB/"sources.json").read_text(encoding="utf-8"))
    manifest=json.loads((LIB/"download_manifest.json").read_text(encoding="utf-8"))
    saved=[r for r in manifest if r["status"]=="downloaded"]
    hashes=[];checks=[];thumbs=[]
    poppler=Path("C:/Users/uBaxa/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe")
    for row in saved:
        path=ROOT/row["file"];h=sha256(path.read_bytes()).hexdigest();assert h==row["sha256"];hashes.append(h)
        if row["id"] in NEW:
            with fitz.open(path) as doc:
                text=re.sub(r"\s+"," ",doc[0].get_text())
                expected=NEW[row["id"]]
                assert expected.lower() in text.lower(),(row["id"],text[:300])
                checks.append(dict(id=row["id"],first_page_title_verified=True,pages=len(doc),sha256=h))
            prefix=LIB/"pages"/(row["id"]+"_v2_p1")
            subprocess.run([str(poppler),"-f","1","-l","1","-singlefile","-r","65","-png",str(path),str(prefix)],check=True)
            image=Image.open(str(prefix)+".png").convert("RGB");image.thumbnail((330,450))
            card=Image.new("RGB",(350,480),"#eceff2");card.paste(image,((350-image.width)//2,8));ImageDraw.Draw(card).text((10,460),row["id"],fill="black");thumbs.append(card)
    sheet=Image.new("RGB",(350*len(thumbs),480),"white")
    for i,image in enumerate(thumbs):
        sheet.paste(image,(350*i,0))
    sheet.save(LIB/"pages/new_papers_v2_contact_sheet.png")
    result=dict(records=len(records),original_pdf_records=len(saved),unique_pdf_sha256=len(set(hashes)),all_saved_pdf_checksums_verified=True,new_first_page_checks=checks,web_documentation_without_original_pdf=[r["id"] for r in records if r.get("role")=="dataset_documentation"],unavailable_pdf=[r["id"] for r in manifest if r["status"]=="unavailable_open_pdf"])
    (LIB/"archive_validation_v2.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))
