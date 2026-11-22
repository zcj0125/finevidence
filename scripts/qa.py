"""End-to-end QA: context from a saved retrieval ranking (or a reference context), expression, execution.

    uv run scripts/qa.py --split dev --contexts bm25 bm25+bge-reranker-v2-m3@30 page gold --top 5

A context name is a ranking file in results/retrieval/<split>/ (top --top chunks, in document order),
or "page" (FinQA's single page) / "gold" (expert evidence only) / "oracle" (the labeled gold chunks of
the filing). The last two are diagnostic ceilings, not systems.
Outputs are cached per question in results/qa/<split>/<name>.jsonl, so reruns only fill gaps.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from fev import PROJECT_ROOT
from fev.data import gold_context, load_proc, page_context
from fev.reason import LLM, is_correct, run_cot, run_direct, run_extract, run_program


def summarize(rows: list[dict]) -> dict:
    return {
        "acc": round(float(np.mean([is_correct(r["pred"], r["gold"]) for r in rows])), 4),
        "acc_lenient": round(float(np.mean([is_correct(r["pred"], r["gold"], lenient_scale=True) for r in rows])), 4),
        "exec_fail": round(float(np.mean([r.get("error") is not None for r in rows])), 4) if "error" in rows[0] else None,
        "tokens_in": round(float(np.mean([r["usage"]["in"] for r in rows])), 1),
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--contexts", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--mode", choices=["program", "cot", "extract", "direct"], default="program")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    exs, docs, gold = load_proc(args.split)
    llm = LLM(args.model)
    fn = {"program": run_program, "cot": run_cot, "extract": run_extract, "direct": run_direct}[args.mode]
    out_dir = PROJECT_ROOT / "results" / "qa" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for name in args.contexts:
        if name == "page":
            ctx = {e.id: page_context(e) for e in exs}
        elif name == "gold":
            ctx = {e.id: gold_context(e) for e in exs}
        elif name == "oracle":  # the labeled gold chunk(s) of the filing; diagnostic only
            ctx = {e.id: "\n\n---\n\n".join(docs[e.doc_id][i] for i in g[:2]) for e, g in zip(exs, gold)}
        else:
            r = json.loads((PROJECT_ROOT / "results" / "retrieval" / args.split / f"{name.replace('/', '_')}.json").read_text())
            assert r["ids"] == [e.id for e in exs]
            ctx = {e.id: "\n\n---\n\n".join(docs[e.doc_id][i] for i in sorted(rank[: args.top]))
                   for e, rank in zip(exs, r["rankings"])}
        tag = f"{name.replace('/', '_')}" + (f".top{args.top}" if name not in ("page", "gold", "oracle") else "") + \
              (f".{args.mode}" if args.mode != "program" else "") + (f".{args.model}" if args.model else "")
        path = out_dir / f"{tag}.jsonl"
        done = {}
        if path.exists():
            for l in path.open():
                d = json.loads(l)
                done[d["id"]] = d
        todo = [e for e in exs if e.id not in done]

        def one(e):
            return {"id": e.id, "gold": e.answer, **fn(llm, ctx[e.id], e.question)}

        with path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
            for row in tqdm(pool.map(one, todo), total=len(todo), desc=tag):
                f.write(json.dumps(row) + "\n")
                done[row["id"]] = row
        summary[tag] = summarize([done[e.id] for e in exs])
        print(tag, summary[tag])
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
