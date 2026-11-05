"""Add the new dataset's primary paper without rewriting historical records."""
import json
from pathlib import Path
from fetch_literature import LIB,archive,write_bib

paper=dict(id="appliances_energy_paper",title="Data driven prediction models of energy use of appliances in a low-energy house",authors="Luis M. Candanedo and Véronique Feldheim and Dominique Deramaix",year=2017,doi="10.1016/j.enbuild.2017.01.083",url="https://orbi.umons.ac.be/handle/20.500.12907/23357",pdf_urls=["https://orbi.umons.ac.be/bitstream/20.500.12907/23357/1/1-s2.0-S0378778816308970-main.pdf"],role="series_v2_dataset_paper",metadata_status="university_repository_and_UCI_verified")
dataset=dict(id="appliances_energy_dataset",title="Appliances Energy Prediction",authors="Luis Candanedo",year=2017,doi="10.24432/C5VC8G",url="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",pdf_urls=[],role="dataset_documentation",metadata_status="primary_UCI_page_verified",original_pdf="not published; web documentation is not a missing journal article")


if __name__=="__main__":
    path=LIB/"sources.json";records=json.loads(path.read_text(encoding="utf-8"))
    ids={r["id"] for r in records}
    records += [r for r in [paper,dataset] if r["id"] not in ids]
    path.write_text(json.dumps(records,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    write_bib(records)
    manifest=LIB/"download_manifest.json";old=json.loads(manifest.read_text(encoding="utf-8"))
    existing={r["id"]:r for r in old};result=archive(paper,existing.get(paper["id"],{}))
    existing[paper["id"]]=result
    manifest.write_text(json.dumps(list(existing.values()),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
