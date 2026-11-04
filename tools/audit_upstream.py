"""Retrieve pinned code for read-only scientific audit, without executing upstream scripts."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REPOS = {"crf": "Camiling/CRF_code", "maxrm": "francescofreni/nldg", "adaxt": "francescofreni/adaXT", "era": "jefferythewind/erasplit", "anchor": "mlondschien/anchorboosting", "trf": "lgmoneda/time-robust-forest"}


def fetch(key, repo):
    headers = {"User-Agent": "coral-forest-dg-research/0.1"}
    response = requests.get(f"https://api.github.com/repos/{repo}/commits/HEAD", headers=headers, timeout=30)
    response.raise_for_status()
    commit = response.json()["sha"]
    response = requests.get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1", headers=headers, timeout=30)
    response.raise_for_status()
    tree = response.json()
    result = dict(repo=repo, commit=commit, tree_truncated=tree.get("truncated"), retrieved_at=datetime.now(timezone.utc).isoformat(), files=[], errors=[])
    for entry in tree.get("tree", []):
        name = entry["path"]
        if entry["type"] != "blob" or entry.get("size", 0) > 250000 or not (name.endswith((".py", ".pyx", ".pxd", ".R", ".r", ".toml", ".txt")) or Path(name).name.lower().startswith(("readme", "license", "copying"))):
            continue
        if any(part in name.lower().split("/") for part in ("data", "datasets", "results", "tests", "test", "benchmarks")):
            continue
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{name}"
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            path = ROOT / "literature/upstream" / key / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            result["files"].append(dict(path=name, sha256=sha256(response.content).hexdigest(), url=url))
        except Exception as error:
            result["errors"].append(dict(path=name, error=str(error)))
    out = ROOT / f"literature/upstream_{key}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return key, commit, len(result["files"]), len(result["errors"])


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(fetch, k, r) for k, r in REPOS.items()]
        for job in as_completed(jobs):
            try:
                print(job.result(), flush=True)
            except Exception as error:
                print(type(error).__name__, str(error), flush=True)
