"""Archive open primary texts; preserve provenance and unsuccessful attempts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import zipfile

import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "literature"
HEADERS = {"User-Agent": "coral-forest-dg-research/0.1 (personal academic literature archive)"}
CORE = [
    dict(id="irf", title="Invariant Random Forest: Tree-Based Model Solution for OOD Generalization", year=2024, authors="Yufan Liao and Qi Wu and Xing Yan", url="https://ojs.aaai.org/index.php/AAAI/article/view/29283", pdf_urls=["https://ojs.aaai.org/index.php/AAAI/article/download/29283/30421", "https://arxiv.org/pdf/2312.04273"], doi="10.1609/aaai.v38i12.29283"),
    dict(id="crf", title="Constrained Random Forest for Domain-Generalizable Classification", year=2026, authors="Camilla Lingjærde and Geir Kjetil Sandve and Arnoldo Frigessi and Sylvia Richardson and Johan Pensar", url="https://proceedings.mlr.press/v337/lingjaerde26a.html", pdf_urls=["https://raw.githubusercontent.com/mlresearch/v337/main/assets/lingjaerde26a/lingjaerde26a.pdf"], doi=""),
    dict(id="meta_forests", title="Meta-forests: Domain generalization on random forests with meta-learning", year=2024, authors="Yuyang Sun and Panagiotis Kosmas", url="https://proceedings.mlr.press/v222/sun24b.html", pdf_urls=["https://proceedings.mlr.press/v222/sun24b/sun24b.pdf", "https://arxiv.org/pdf/2401.04425"], doi=""),
    dict(id="maxrm_rf", title="Maximum Risk Minimization with Random Forests", year=2026, authors="Francesco Freni and Anya Fries and Linus Kühne and Markus Reichstein and Jonas Peters", url="https://arxiv.org/abs/2512.10445", pdf_urls=["https://arxiv.org/pdf/2512.10445v2"], doi="10.48550/arXiv.2512.10445", version="v2, 2026-03-11"),
    dict(id="era_splitting", title="Era splitting - Invariant learning for decision trees", year=2026, authors="Timothy DeLise", url="https://doi.org/10.1016/j.eswa.2025.129687", pdf_urls=["https://arxiv.org/pdf/2309.14496"], doi="10.1016/j.eswa.2025.129687", version="open arXiv copy; compare with journal version"),
    dict(id="anchor_boosting", title="Domain generalization and adaptation in intensive care with anchor regression", year=2026, authors="Michael Londschien and Michael Burger and Gunnar Rätsch and Peter Bühlmann", url="https://doi.org/10.1093/rssdat/udag001", pdf_urls=["https://arxiv.org/pdf/2507.21783"], doi="10.1093/rssdat/udag001", version="open arXiv copy; compare with journal version"),
]
EXTRA = [
    ("time_robust_trees", "Time Robust Trees: Using Temporal Invariance to Improve Generalization", "https://lgmoneda.github.io/resources/papers/Time_Robust_Tree.pdf"),
    ("mixmax", "MixMax: Distributional Robustness in Function Space via Optimal Data Mixtures", "https://arxiv.org/pdf/2406.01477"),
    ("tree_dro", "DRO: A Python Library for Distributionally Robust Optimization in Machine Learning", "https://arxiv.org/pdf/2505.23565"),
    ("bps", "Boosting for Predictive Sufficiency", "https://openreview.net/pdf?id=1mQT8PXIy8"),
    ("domainbed", "In Search of Lost Domain Generalization", "https://arxiv.org/pdf/2007.01434"),
    ("rf", "Random Forests", "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf"),
    ("gbm", "Greedy Function Approximation: A Gradient Boosting Machine", "https://jerryfriedman.su.domains/ftp/trebst.pdf"),
    ("lightgbm", "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", "https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf"),
    ("catboost", "CatBoost: unbiased boosting with categorical features", "https://arxiv.org/pdf/1706.09516"),
    ("xgboost", "XGBoost: A Scalable Tree Boosting System", "https://arxiv.org/pdf/1603.02754"),
    ("demsar", "Statistical Comparisons of Classifiers over Multiple Data Sets", "https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf"),
    ("anchor_regression", "Anchor Regression: Heterogeneous Data Meet Causality", "https://arxiv.org/pdf/1801.06229"),
    ("trees_tabular", "Why do tree-based models still outperform deep learning on typical tabular data?", "https://arxiv.org/pdf/2207.08815"),
]


def handoff_records():
    doc = ROOT / "research_input/NIR_Codex_Handoff_v2/context/02_initial_literature_review.docx"
    with zipfile.ZipFile(doc) as z:
        xml = ET.fromstring(z.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = ["".join(t.text or "" for t in p.findall(".//w:t", ns)) for p in xml.findall(".//w:body//w:p", ns)]
    records = []
    for i, text in enumerate(paragraphs):
        match = re.match(r"\[(\d+)\]\s+(.+)", text)
        if not match:
            continue
        number, citation = match.groups()
        after = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
        urls = re.findall(r"https?://\S+", after)
        pdfs = []
        for arxiv in re.findall(r"arXiv[:\s]+(\d{4}\.\d{4,5})", citation):
            pdfs.append("https://arxiv.org/pdf/" + arxiv)
        for volume, paper in re.findall(r"PMLR\s+v(\d+)/(\w+)", citation):
            pdfs.append(f"https://proceedings.mlr.press/v{volume}/{paper}/{paper}.pdf")
        for u in urls:
            if u.endswith(".pdf"):
                pdfs.append(u)
        records.append(dict(id=f"handoff_{int(number):02d}", title=citation, citation_raw=citation, url=urls[0] if urls else "", pdf_urls=pdfs, role="historical_candidate", metadata_status="imported_not_independently_verified"))
    return records


def init_registry():
    path = LIB / "sources.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    records = [dict(x, role="core_method", metadata_status="primary_landing_page_checked") for x in CORE]
    records += [dict(id=i, title=t, url=u, pdf_urls=[u], role="supporting_source", metadata_status="PDF_pending_validation") for i, t, u in EXTRA]
    records += handoff_records()
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


def discover_pdf(record):
    url = record.get("url")
    if not url:
        return []
    response = requests.get(url, headers=HEADERS, timeout=(10, 25))
    response.raise_for_status()
    html = response.text
    links = re.findall(r'<meta[^>]+name=[\'"]citation_pdf_url[\'"][^>]+content=[\'"]([^\'"]+)', html, flags=re.I)
    links += re.findall(r'<a[^>]+href=[\'"]([^\'"]+\.pdf(?:\?[^\'"]*)?)[\'"]', html, flags=re.I)
    return [urljoin(response.url, link.replace("&amp;", "&")) for link in links[:3]]


def archive(record, previous):
    identifier = record["id"]
    path = LIB / "pdfs" / f"{identifier}.pdf"
    if path.exists() and previous.get("sha256") == sha256(path.read_bytes()).hexdigest():
        return previous
    attempts = []
    urls = list(record.get("pdf_urls", []))
    if not urls:
        try:
            urls = discover_pdf(record)
        except Exception as error:
            attempts.append({"url": record.get("url"), "error": str(error)})
    for url in dict.fromkeys(urls):
        try:
            response = requests.get(url, headers=HEADERS, timeout=(10, 40))
            response.raise_for_status()
            content = response.content
            if not content.startswith(b"%PDF-"):
                raise ValueError("Response is not a PDF (HTML/paywall is not archived as full text)")
            reader = PdfReader(io.BytesIO(content))
            texts = [f"\n\n--- PDF PAGE {i + 1} ---\n" + (p.extract_text() or "") for i, p in enumerate(reader.pages)]
            if not reader.pages:
                raise ValueError("PDF has no readable pages")
            path.write_bytes(content)
            (LIB / "text" / f"{identifier}.txt").write_text("".join(texts), encoding="utf-8")
            return dict(id=identifier, status="downloaded", file=path.relative_to(ROOT).as_posix(), sha256=sha256(content).hexdigest(), size_bytes=len(content), pages=len(reader.pages), url=url, resolved_url=response.url, retrieved_at=datetime.now(timezone.utc).isoformat(), attempts=attempts, content_identity="requires_reader_verification", redistribution="not_assessed_local_archive")
        except Exception as error:
            attempts.append({"url": url, "error": str(error)})
    return dict(id=identifier, status="unavailable_open_pdf", attempts=attempts, checked_at=datetime.now(timezone.utc).isoformat())


def write_bib(records):
    entries = []
    for r in records:
        fields = {k: r[k] for k in ("title", "authors", "year", "url", "doi") if r.get(k)}
        if "authors" in fields:
            fields["author"] = fields.pop("authors")
        body = ",\n".join(f"  {k} = {{{str(v).replace('{', '').replace('}', '')}}}" for k, v in fields.items())
        entries.append(f"@misc{{{r['id']},\n{body}\n}}")
    (LIB / "references.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for directory in ("pdfs", "text", "pages", "upstream"):
        (LIB / directory).mkdir(parents=True, exist_ok=True)
    records = init_registry()
    write_bib(records)
    manifest = LIB / "download_manifest.json"
    old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    results = {r["id"]: r for r in old}
    selected = [r for r in records if args.ids is None or r["id"] in args.ids]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(archive, record, results.get(record["id"], {})): record for record in selected}
        for job in as_completed(jobs):
            result = job.result()
            results[result["id"]] = result
            manifest.write_text(json.dumps(list(results.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(result["id"], result["status"], result.get("pages", ""), flush=True)
    print(f"Archived {sum(r['status'] == 'downloaded' for r in results.values())}/{len(results)} records", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
