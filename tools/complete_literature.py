"""Add verified open alternatives and dataset papers to the retained source registry."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    path = ROOT / "literature/sources.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in records}
    aliases = {"handoff_01": "irf", "handoff_02": "crf", "handoff_03": "era_splitting", "handoff_04": "time_robust_trees", "handoff_05": "anchor_boosting", "handoff_11": "anchor_regression", "handoff_33": "domainbed", "handoff_35": "trees_tabular", "handoff_36": "demsar", "handoff_37": "rf", "handoff_38": "gbm", "handoff_39": "xgboost", "handoff_40": "lightgbm", "handoff_41": "catboost"}
    for alias, canonical in aliases.items():
        by_id[alias]["alias_of"] = canonical
        by_id[alias]["pdf_urls"] = list(dict.fromkeys(by_id[canonical]["pdf_urls"] + by_id[alias].get("pdf_urls", [])))
    by_id["bps"]["pdf_urls"] += ["https://proceedings.iclr.cc/paper_files/paper/2026/file/4f1c0a593705124f3061ca7d48b92898-Paper-Conference.pdf"]
    by_id["handoff_42"]["pdf_urls"] += ["https://cs.uwaterloo.ca/~brecht/courses/854-http-video-2012/readings/keshav-paper-reading.pdf"]
    additions = [
        dict(id="gas_turbine_pems", title="Predicting CO and NOx emissions from gas turbines: novel data and a benchmark PEMS", authors="Heysem Kaya and Pınar Tüfekci and Erdinç Uzun", year=2019, doi="10.3906/elk-1807-87", url="https://research-portal.uu.nl/en/publications/predicting-co-and-nox-emissions-from-gas-turbines-novel-data-and-/", pdf_urls=["https://dbc.library.uu.nl/bitstream/handle/1874/407781/elk_27_6_54_1807_87.pdf?isAllowed=y&sequence=1"], role="used_dataset_paper", metadata_status="university_record_checked"),
        dict(id="beijing_air_quality", title="Cautionary tales on air-quality improvement in Beijing", authors="Shuyi Zhang and Bin Guo and Anlan Dong and Jing He and Ziping Xu and Song Xi Chen", year=2017, doi="10.1098/rspa.2017.0457", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5627385/", pdf_urls=["https://pmc.ncbi.nlm.nih.gov/articles/PMC5627385/pdf/rspa20170457.pdf", "https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2017.0457"], role="used_dataset_paper", metadata_status="PubMed_record_checked"),
        dict(id="hospital_diabetes_paper", title="Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records", year=2014, doi="10.1155/2014/781670", url="https://doi.org/10.1155/2014/781670", pdf_urls=["https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/pdf/BMRI2014-781670.pdf"], role="used_dataset_paper", metadata_status="TableShift_primary_link_checked_PDF_identity_to_verify"),
    ]
    for record in additions:
        if record["id"] not in by_id:
            records.append(record)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Registry contains {len(records)} records, including retained historical candidates")
