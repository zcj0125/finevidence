"""Pilot: where is the headroom? Same generator (gpt-4.1-mini, temperature 0) under different contexts.

  page_direct   FinQA page (~680 words), answer directly
  page_program  FinQA page, write an expression we execute
  gold_program  only the expert-annotated evidence facts (diagnostic ceiling, not a system)
  doc_<r>_top5  full 10-K (DocFinQA, ~86k words) chunked, top-5 chunks from retriever r, expression

Also reports chunk retrieval hit@k (gold chunk = chunk holding the gold facts' numbers) on all labeled dev
questions. Dev split only.

    uv run scripts/pilot.py --n 200
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from fev import PROJECT_ROOT
from fev.chunk import chunk_document, label_gold_chunks
from fev.data import gold_context, load_docfinqa, page_context
from fev.reason import LLM, is_correct, run_direct, run_program

OUT = PROJECT_ROOT / "results" / "pilot"
CACHE = PROJECT_ROOT / "cache"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def pick_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def bm25_rank(chunks_by_doc, exs, k):
    import bm25s
    import Stemmer

    stem = Stemmer.Stemmer("english")
    idx = {}
    for d in sorted({e.doc_id for e in exs}):
        r = bm25s.BM25()
        r.index(bm25s.tokenize(chunks_by_doc[d], stopwords="en", stemmer=stem, show_progress=False), show_progress=False)
        idx[d] = r
    out = {}
    for e in exs:
        n = len(chunks_by_doc[e.doc_id])
        res, _ = idx[e.doc_id].retrieve(bm25s.tokenize([e.question], stopwords="en", stemmer=stem, show_progress=False),
                                        k=min(k, n), show_progress=False)
        out[e.id] = [int(i) for i in res[0]]
    return out


def dense_rank(chunks_by_doc, exs, k, model_name="BAAI/bge-small-en-v1.5"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=pick_device())
    model.max_seq_length = 512
    docs = sorted({e.doc_id for e in exs})
    cache = CACHE / f"chunks_{model_name.replace('/', '_')}.npz"
    emb = dict(np.load(cache)) if cache.exists() else {}
    todo = [d for d in docs if str(d) not in emb]
    if todo:
        flat = [(d, c) for d in todo for c in chunks_by_doc[d]]
        vecs = model.encode([c for _, c in flat], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        i = 0
        for d in todo:
            n = len(chunks_by_doc[d])
            emb[str(d)] = vecs[i:i + n]
            i += n
        CACHE.mkdir(exist_ok=True)
        np.savez(cache, **emb)
    q = model.encode([QUERY_PREFIX + e.question for e in exs], normalize_embeddings=True)
    return {e.id: [int(i) for i in np.argsort(-(emb[str(e.doc_id)] @ qv))[:k]] for e, qv in zip(exs, q)}


def rerank(chunks_by_doc, exs, first, k, teacher="BAAI/bge-reranker-v2-m3"):
    import torch
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(teacher, device=pick_device(), activation_fn=torch.nn.Identity(), max_length=512)
    pairs = [(e.question, chunks_by_doc[e.doc_id][i]) for e in exs for i in first[e.id][:k]]
    scores = iter(ce.predict(pairs, batch_size=16, show_progress_bar=True))
    out = {}
    for e in exs:
        cand = first[e.id][:k]
        s = [float(next(scores)) for _ in cand]
        out[e.id] = [cand[j] for j in np.argsort(s)[::-1]]
    return out


def hits(ranking, gold, ks=(1, 3, 5, 10, 20)):
    ids = [i for i in ranking if gold.get(i)]
    return {f"hit@{k}": float(np.mean([bool(set(ranking[i][:k]) & set(gold[i])) for i in ids])) for k in ks} | {"n": len(ids)}


def run_config(name, exs, make_context, fn, llm, workers=16):
    path = OUT / f"{name}.jsonl"
    done = {json.loads(l)["id"]: json.loads(l) for l in path.open()} if path.exists() else {}
    todo = [e for e in exs if e.id not in done]

    def one(e):
        return {"id": e.id, "gold": e.answer, **fn(llm, make_context(e), e.question)}

    with path.open("a") as f, ThreadPoolExecutor(workers) as ex:
        for r in tqdm(ex.map(one, todo), total=len(todo), desc=name):
            f.write(json.dumps(r) + "\n")
            done[r["id"]] = r
    rows = [done[e.id] for e in exs]
    return {
        "acc": float(np.mean([is_correct(r["pred"], r["gold"]) for r in rows])),
        "acc_lenient": float(np.mean([is_correct(r["pred"], r["gold"], lenient_scale=True) for r in rows])),
        "exec_fail": float(np.mean([r.get("error") is not None for r in rows])) if "error" in rows[0] else None,
        "tokens_in": float(np.mean([r["usage"]["in"] for r in rows])),
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    exs, docs = load_docfinqa("dev")
    chunks_by_doc = {d: chunk_document(t) for d, t in enumerate(docs)}
    gold = {e.id: label_gold_chunks(chunks_by_doc[e.doc_id], e.gold_inds) for e in exs}
    labeled = [e for e in exs if gold[e.id]]
    sample = random.Random(args.seed).sample(labeled, args.n)

    report = {"n_dev": len(exs), "n_labeled": len(labeled), "retrieval": {}, "qa": {}}
    bm = bm25_rank(chunks_by_doc, labeled, 50)
    dn = dense_rank(chunks_by_doc, labeled, 50)
    report["retrieval"]["bm25"] = hits(bm, gold)
    report["retrieval"]["bge-small"] = hits(dn, gold)
    rr = rerank(chunks_by_doc, sample, dn, 30)
    report["retrieval"]["bge-small>v2-m3 (sample)"] = hits(rr, {e.id: gold[e.id] for e in sample})
    report["retrieval"]["bge-small (sample)"] = hits({e.id: dn[e.id] for e in sample}, {e.id: gold[e.id] for e in sample})
    print(json.dumps(report["retrieval"], indent=1))

    def top_chunks(rank):
        # keep document order so tables read naturally
        return lambda e: "\n\n---\n\n".join(chunks_by_doc[e.doc_id][i] for i in sorted(rank[e.id][:args.top]))

    llm = LLM()
    configs = {
        "page_direct": (page_context, run_direct),
        "page_program": (page_context, run_program),
        "gold_program": (gold_context, run_program),
        f"doc_bm25_top{args.top}": (top_chunks(bm), run_program),
        f"doc_dense_top{args.top}": (top_chunks(dn), run_program),
        f"doc_rerank_top{args.top}": (top_chunks(rr), run_program),
    }
    for name, (ctx, fn) in configs.items():
        report["qa"][name] = run_config(name, sample, ctx, fn, llm)
        print(name, report["qa"][name])
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
