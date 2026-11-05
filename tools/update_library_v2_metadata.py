"""Record verified open versions; retain every imported citation_raw unchanged."""
import json
from pathlib import Path
from fetch_literature import LIB,write_bib

UPDATES={
    "handoff_06":dict(title="Spectrally Deconfounded Random Forests",authors="Markus Ulmer and Cyrill Scheidegger and Peter Bühlmann",year=2025,url="https://arxiv.org/abs/2502.03969",doi="10.48550/arXiv.2502.03969",published_doi="10.1080/10618600.2025.2569602",version="arXiv v3, 2025-09-24; journal citation remains in citation_raw"),
    "handoff_17":dict(title="Decorr: Environment Partitioning for Invariant Learning and OOD Generalization",authors="Yufan Liao and Qi Wu and Zhaodi Wu and Xing Yan",year=2024,url="https://arxiv.org/abs/2211.10054",doi="10.48550/arXiv.2211.10054",version="arXiv v2, 2024-05-22; original later journal reference not substituted for this PDF"),
    "handoff_22":dict(title="On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets",authors="Jiashuo Liu and Tianyu Wang and Peng Cui and Hongseok Namkoong",year=2023,url="https://proceedings.neurips.cc/paper_files/paper/2023/hash/a134eaebd55b7406ff29cd75d5f1a622-Abstract.html",doi="10.52202/075280-2237",version="NeurIPS 2023 proceedings PDF, including appendix"),
    "handoff_31":dict(title="TabFSBench: Tabular Benchmark for Feature Shifts in Open Environments",authors="Zi-Jian Cheng and Zi-Yi Jia and Zhi Zhou and Yu-Feng Li and Lan-Zhe Guo",year=2025,url="https://arxiv.org/abs/2501.18935",doi="10.48550/arXiv.2501.18935",version="arXiv v3, 2025-05-31; PMLR reference retained in citation_raw"),
}


if __name__=="__main__":
    p=LIB/"sources.json";records=json.loads(p.read_text(encoding="utf-8"))
    for r in records:
        if r["id"] in UPDATES:
            r.update(UPDATES[r["id"]]);r["metadata_status"]="primary_page_and_original_PDF_first_page_verified"
    p.write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");write_bib(records)
    p=LIB/"download_manifest.json";manifest=json.loads(p.read_text(encoding="utf-8"))
    for r in manifest:
        if r["id"] in {*UPDATES,"appliances_energy_paper"}:
            r["content_identity"]="first_page_title_and_authors_verified_text_and_visual"
    p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    p=LIB/"archive_validation_v2.json";v=json.loads(p.read_text(encoding="utf-8"));v["new_first_pages_visually_reviewed"]=True
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
