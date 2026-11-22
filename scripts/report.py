"""Tables for the README: retrieval hit@k, QA accuracy, and paired-bootstrap deltas.

    uv run scripts/report.py --split test
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import numpy as np

from fev import PROJECT_ROOT
from fev.data import load_proc
from fev.reason import is_correct

RETRIEVAL = ["bm25", "dense", "rrf", "bm25+bge-reranker-base@30", "bm25+bge-reranker-v2-m3@30", "bm25+reranker-base-ft@30"]
QA = ["gold", "page", "page.direct", "oracle", "bm25.top5", "bm25+bge-reranker-v2-m3@30.top3", "bm25+reranker-base-ft@30.top3"]
PAIRS = [("page", "page.direct"), ("bm25.top5", "bm25+reranker-base-ft@30.top3"),
         ("bm25+bge-reranker-v2-m3@30.top3", "bm25+reranker-base-ft@30.top3")]


def bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0):
    d = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    exs, _, gold = load_proc(args.split)
    lab = np.array([len(g) > 0 for g in gold])
    rdir = PROJECT_ROOT / "results" / "retrieval" / args.split
    qdir = PROJECT_ROOT / "results" / "qa" / args.split

    print(f"## {args.split}: {len(exs)} questions, {lab.sum()} with a labeled gold chunk\n")
    print("| retriever | hit@1 | hit@3 | hit@5 | hit@10 | hit@30 |\n|---|---|---|---|---|---|")
    ranks = {}
    for m in RETRIEVAL:
        p = rdir / f"{m}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        ranks[m] = r["rankings"]
        h = r["metrics"]
        print(f"| {m} | " + " | ".join(f"{h[f'hit@{k}']:.3f}" for k in (1, 3, 5, 10, 30)) + " |")

    correct = {}
    print("\n| context | accuracy | lenient (x100) | exec fail | input tokens |\n|---|---|---|---|---|")
    for t in QA:
        p = qdir / f"{t}.jsonl"
        if not p.exists():
            continue
        rows = {d["id"]: d for d in map(json.loads, p.open())}
        rows = [rows[e.id] for e in exs]
        c = np.array([is_correct(r["pred"], r["gold"]) for r in rows])
        correct[t] = c
        cl = np.mean([is_correct(r["pred"], r["gold"], lenient_scale=True) for r in rows])
        ef = np.mean([r.get("error") is not None for r in rows]) if "error" in rows[0] else float("nan")
        tok = np.mean([r["usage"]["in"] for r in rows])
        print(f"| {t} | {c.mean():.3f} | {cl:.3f} | {ef:.3f} | {tok:.0f} |")

    print("\n| comparison (B - A) | delta | 95% CI |\n|---|---|---|")
    for a, b in PAIRS:
        if a in correct and b in correct:
            d, lo, hi = bootstrap(correct[a], correct[b])
            print(f"| {b} vs {a} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")

    t = "bm25+reranker-base-ft@30.top3"
    if t in correct and "bm25+reranker-base-ft@30" in ranks:
        hit = np.array([bool(set(r[:3]) & set(g)) for r, g in zip(ranks["bm25+reranker-base-ft@30"], gold)])
        c = correct[t]
        print(f"\nError decomposition ({t}):")
        for name, m in [("gold chunk in top 3", lab & hit), ("labeled, missed", lab & ~hit), ("unlabeled", ~lab)]:
            print(f"- {name}: n={m.sum()}, accuracy {c[m].mean():.3f}")
        if "oracle" in correct:
            print(f"- oracle chunk accuracy on labeled questions: {correct['oracle'][lab].mean():.3f}")


if __name__ == "__main__":
    main()
