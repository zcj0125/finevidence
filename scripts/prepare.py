"""Join DocFinQA with FinQA, chunk each unique filing once, label gold chunks.

Writes data/proc/<split>.docs.jsonl     one JSON list of chunk strings per line (line number = doc_id)
       data/proc/<split>.examples.jsonl one example per line, with doc_id and gold_chunks

    uv run scripts/prepare.py --splits dev test train
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import gc
import json

from fev.chunk import chunk_document, label_gold_chunks
from fev.data import DATA, load_docfinqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "train"])
    args = ap.parse_args()
    out = DATA / "proc"
    out.mkdir(exist_ok=True)
    for split in args.splits:
        exs, docs = load_docfinqa(split)
        chunks = [chunk_document(d) for d in docs]
        del docs
        gc.collect()
        with (out / f"{split}.docs.jsonl").open("w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        n_lab = 0
        with (out / f"{split}.examples.jsonl").open("w") as f:
            for e in exs:
                g = label_gold_chunks(chunks[e.doc_id], e.gold_inds)
                n_lab += bool(g)
                f.write(json.dumps({**e.__dict__, "gold_chunks": g}) + "\n")
        print(f"{split}: {len(exs)} examples, {len(chunks)} docs, "
              f"{sum(map(len, chunks))} chunks, {n_lab} with gold chunks")


if __name__ == "__main__":
    main()
