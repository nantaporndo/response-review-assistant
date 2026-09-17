"""Retrieval evaluation: recall@k / precision@k / MRR ต่อ query แล้วเฉลี่ย
 
Test set format (eval/test_set.yaml):
  - query: "refund never arrived"
    relevant_ids: ["123", "456"]   # review_id ที่ "ควร" ถูกดึงขึ้นมา
 
หมายเหตุเรื่อง ground truth: relevant_ids ไม่จำเป็นต้องครบทุก doc ที่เกี่ยว
(corpus 21K แถว label หมดไม่ไหว) -> recall@k ที่ได้คือ "recall ของชุดที่รู้"
ยิ่ง label ครบ ตัวเลขยิ่งน่าเชื่อ — เป็น limitation ที่ต้องเขียนใน README
"""

from pathlib import Path

import yaml

from src.config import PROJECT_ROOT,load_config
from src.retrieve import search
from src.vectorstore import get_collection

def load_test_set(path:str | Path | None = None, config:dict | None = None) -> list[dict]:
    cfg = config or load_config()
    p = Path(path) if path else PROJECT_ROOT / cfg["evaluation"]["test_set_path"]
    with open(p,encoding="uft-8") as f:
        return yaml.safe_load(f)

def evaluate_query(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> dict:
    top_k = retrieved_ids[:k]
    hits = [rid for rid in top_k if rid in relevant_ids]
    recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0
    precision = len(hits) / k
    rr = 0.0
    for rank, rid in enumerate(top_k, start=1):
        if rid in relevant_ids:
            rr = 1.0 / rank
            break
    return {"recall": recall, "precision": precision, "rr": rr}
 
 
def run(config: dict | None = None, verbose: bool = True) -> dict:
    cfg = config or load_config()
    test_set = load_test_set(config=cfg)
    k_values = cfg["evaluation"]["k_values"]
    max_k = max(k_values)
    collection = get_collection(cfg)
 
    per_k = {k: {"recall": [], "precision": [], "rr": []} for k in k_values}
 
    for case in test_set:
        results = search(case["query"], k=max_k, config=cfg, collection=collection)
        retrieved = [r.review_id for r in results]
        relevant = [str(x) for x in case["relevant_ids"]]
        for k in k_values:
            m = evaluate_query(retrieved, relevant, k)
            for name, val in m.items():
                per_k[k][name].append(val)
        if verbose:
            m5 = evaluate_query(retrieved, relevant, min(5, max_k))
            print(f"recall@5={m5['recall']:.2f}  {case['query'][:60]!r}")
 
    summary = {
        f"{name}@{k}": round(sum(vals) / len(vals), 3)
        for k in k_values
        for name, vals in per_k[k].items()
        if vals
    }
    if verbose:
        print("\n=== summary (mean over", len(test_set), "queries) ===")
        for key in sorted(summary):
            print(f"{key:>14}: {summary[key]}")
    return summary