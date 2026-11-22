"""Reranker training pairs from a split's gold chunk labels and saved first-stage rankings.

Positives: the gold chunks (up to --max-pos). Negatives: the highest-ranked first-stage chunks that are not
gold and hold less than --fn-frac of the gold numbers (a chunk with most of the evidence numbers is likely a
duplicate table, i.e. an unlabeled positive). These are the hard cases: same filing, same years, near-miss
tables. A few random chunks from the same filing are added so the model still sees easy negatives.

    uv run scripts/build_ce_data.py --split train --first rrf --n-hard 7 --n-random 1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import random

from fev import PROJECT_ROOT
from fev.chunk import _is_year, numbers
from fev.data import DATA, load_proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--first", default="rrf")
    ap.add_argument("--n-hard", type=int, default=7)
    ap.add_argument("--n-random", type=int, default=1)
    ap.add_argument("--max-pos", type=int, default=2)
    ap.add_argument("--hard-depth", type=int, default=30)
    ap.add_argument("--fn-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to train on test")
    rng = random.Random(args.seed)

    exs, docs, gold = load_proc(args.split)
    ranking = json.loads((PROJECT_ROOT / "results" / "retrieval" / args.split / f"{args.first}.json").read_text())
    assert ranking["ids"] == [e.id for e in exs]

    out = DATA / "proc" / f"{args.out or f'{args.split}.ce.{args.first}'}.jsonl"
    n_q = n_pairs = n_fn = 0
    with out.open("w") as f:
        for e, g, rank in zip(exs, gold, ranking["rankings"]):
            if not g:
                continue
            chunks = docs[e.doc_id]
            gnum = {x for x in numbers(" ".join(e.gold_inds.values())) if not _is_year(x)} or set()
            def likely_pos(i):
                return bool(gnum) and len(gnum & numbers(chunks[i])) / len(gnum) >= args.fn_frac
            hard = []
            for i in rank[: args.hard_depth]:
                if i in g:
                    continue
                if likely_pos(i):
                    n_fn += 1
                    continue
                hard.append(i)
            hard = hard[: args.n_hard]
            pool = [i for i in range(len(chunks)) if i not in g and i not in hard]
            easy = rng.sample(pool, min(args.n_random, len(pool)))
            for p in g[: args.max_pos]:
                f.write(json.dumps({"query": e.question, "doc": chunks[p], "label": 1}) + "\n")
                n_pairs += 1
            for i in hard + easy:
                f.write(json.dumps({"query": e.question, "doc": chunks[i], "label": 0}) + "\n")
                n_pairs += 1
            n_q += 1
    print(f"{n_pairs} pairs from {n_q} questions -> {out.name}; skipped {n_fn} likely-positive negatives")


if __name__ == "__main__":
    main()
