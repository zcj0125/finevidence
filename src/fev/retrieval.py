"""Chunk retrieval inside one filing: BM25, dense, reciprocal-rank fusion, cross-encoder reranking.

Every question is asked about a known filing (the user picks the 10-K), so each ranking is over that
document's chunks only. Rankings are lists of chunk indices, best first.
"""

from __future__ import annotations

import numpy as np

from fev import PROJECT_ROOT

CACHE = PROJECT_ROOT / "cache"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def bm25_rank(docs: list[list[str]], queries: list[tuple[int, str]], k: int = 100) -> list[list[int]]:
    """queries: (doc_id, text). One BM25 index per document, built lazily."""
    import bm25s
    import Stemmer

    stem = Stemmer.Stemmer("english")
    tok = lambda xs: bm25s.tokenize(xs, stopwords="en", stemmer=stem, show_progress=False)  # noqa: E731
    idx: dict[int, bm25s.BM25] = {}
    out = []
    for d, q in queries:
        if d not in idx:
            idx[d] = bm25s.BM25()
            idx[d].index(tok(docs[d]), show_progress=False)
        res, _ = idx[d].retrieve(tok([q]), k=min(k, len(docs[d])), show_progress=False)
        out.append([int(i) for i in res[0]])
    return out


class Dense:
    def __init__(self, name: str = "BAAI/bge-small-en-v1.5", query_prefix: str = BGE_QUERY_PREFIX):
        from sentence_transformers import SentenceTransformer

        self.name, self.query_prefix = name, query_prefix
        self.model = SentenceTransformer(name, device=pick_device())
        self.model.max_seq_length = 512

    def doc_embeddings(self, docs: list[list[str]], tag: str) -> list[np.ndarray]:
        """Embeddings of every chunk, cached per (tag, model) as one flat array plus offsets."""
        path = CACHE / f"{tag}.{self.name.replace('/', '_')}.npz"
        sizes = np.array([len(c) for c in docs])
        if path.exists():
            z = np.load(path)
            if np.array_equal(z["sizes"], sizes):
                flat = z["emb"]
                return np.split(flat, np.cumsum(sizes)[:-1])
        flat = self.model.encode([c for doc in docs for c in doc], batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=True)
        CACHE.mkdir(exist_ok=True)
        np.savez(path, emb=flat, sizes=sizes)
        return np.split(flat, np.cumsum(sizes)[:-1])

    def rank(self, doc_emb: list[np.ndarray], queries: list[tuple[int, str]], k: int = 100) -> list[list[int]]:
        q = self.model.encode([self.query_prefix + t for _, t in queries], normalize_embeddings=True, batch_size=64)
        return [[int(i) for i in np.argsort(-(doc_emb[d] @ qv))[:k]] for (d, _), qv in zip(queries, q)]


def rrf(*rankings: list[list[int]], k: int = 60, top: int = 100) -> list[list[int]]:
    """Reciprocal-rank fusion of several rankings per query."""
    out = []
    for per_q in zip(*rankings):
        score: dict[int, float] = {}
        for r in per_q:
            for pos, i in enumerate(r):
                score[i] = score.get(i, 0.0) + 1.0 / (k + pos + 1)
        out.append(sorted(score, key=score.get, reverse=True)[:top])
    return out


def load_cross_encoder(name: str, max_length: int = 512):
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device=pick_device(), activation_fn=torch.nn.Identity(), max_length=max_length)


def rerank(ce, docs: list[list[str]], queries: list[tuple[int, str]], first: list[list[int]], k: int = 30,
           batch_size: int = 16) -> list[list[int]]:
    pairs = [(q, docs[d][i]) for (d, q), cand in zip(queries, first) for i in cand[:k]]
    scores = iter(ce.predict(pairs, batch_size=batch_size, show_progress_bar=True))
    out = []
    for cand in first:
        cand = cand[:k]
        s = np.array([float(next(scores)) for _ in cand])
        out.append([cand[j] for j in np.argsort(-s)])
    return out


def hit_at(rankings: list[list[int]], gold: list[list[int]], ks=(1, 3, 5, 10, 20, 30)) -> dict:
    keep = [(r, g) for r, g in zip(rankings, gold) if g]
    return {f"hit@{k}": round(float(np.mean([bool(set(r[:k]) & set(g)) for r, g in keep])), 4) for k in ks} | {
        "n": len(keep)}
