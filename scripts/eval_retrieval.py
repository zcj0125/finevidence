"""Chunk retrieval on a split: hit@k against gold chunks. Saves every ranking for the QA step.

    uv run scripts/eval_retrieval.py --split dev --methods bm25 dense rrf rrf+BAAI/bge-reranker-v2-m3
    uv run scripts/eval_retrieval.py --split dev --methods rrf+models/reranker-ft

A method "<first>+<cross-encoder>" reranks the top --rerank-k of <first> (a saved ranking).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

from fev import PROJECT_ROOT
from fev.data import load_proc
from fev.retrieval import Dense, bm25_rank, hit_at, load_cross_encoder, rerank, rrf


def slug(name: str) -> str:
    return name.replace("/", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--methods", nargs="+", default=["bm25", "dense", "rrf"])
    ap.add_argument("--rerank-k", type=int, default=30)
    ap.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()
    if args.split == "test":
        print("note: evaluating on test")

    exs, docs, gold = load_proc(args.split)
    queries = [(e.doc_id, e.question) for e in exs]
    out = PROJECT_ROOT / "results" / "retrieval" / args.split
    out.mkdir(parents=True, exist_ok=True)

    def saved(name):
        p = out / f"{slug(name)}.json"
        return json.loads(p.read_text())["rankings"] if p.exists() else None

    def save(name, rankings):
        m = hit_at(rankings, gold)
        (out / f"{slug(name)}.json").write_text(json.dumps({"method": name, "metrics": m, "ids": [e.id for e in exs],
                                                           "rankings": rankings}))
        print(f"{name:45s} " + "  ".join(f"{k}={v}" for k, v in m.items()))

    for method in args.methods:
        if method == "bm25":
            save("bm25", bm25_rank(docs, queries))
        elif method == "dense":
            dense = Dense(args.dense_model)
            save("dense", dense.rank(dense.doc_embeddings(docs, f"{args.split}"), queries))
        elif method == "rrf":
            b, d = saved("bm25"), saved("dense")
            assert b and d, "run bm25 and dense first"
            save("rrf", rrf(b, d))
        elif "+" in method:
            first, model = method.split("+", 1)
            base = saved(first)
            assert base, f"run {first} first"
            ce = load_cross_encoder(model)
            save(f"{first}+{Path(model).name}@{args.rerank_k}", rerank(ce, docs, queries, base, k=args.rerank_k))
        else:
            raise SystemExit(f"unknown method {method}")


if __name__ == "__main__":
    main()
