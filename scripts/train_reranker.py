"""Fine-tune a cross-encoder on (question, chunk, label) pairs with binary cross-entropy.

    uv run scripts/train_reranker.py --data train.ce.rrf --base BAAI/bge-reranker-base --out reranker-base-ft
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import torch

from fev import PROJECT_ROOT
from fev.data import DATA
from fev.retrieval import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train.ce.rrf")
    ap.add_argument("--base", default="BAAI/bge-reranker-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N pairs (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    rows = [json.loads(l) for l in (DATA / "proc" / f"{args.data}.jsonl").open()]
    if args.limit:
        rows = rows[: args.limit]
    ds = Dataset.from_dict({"query": [r["query"] for r in rows], "doc": [r["doc"] for r in rows],
                            "label": [float(r["label"]) for r in rows]}).shuffle(seed=args.seed)
    n_pos = sum(r["label"] for r in rows)
    print(f"{len(rows)} pairs, {n_pos} positive")

    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length, device=pick_device())
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor((len(rows) - n_pos) / max(n_pos, 1)))
    out = PROJECT_ROOT / "models" / args.out
    targs = CrossEncoderTrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, warmup_ratio=0.1, logging_steps=50, save_strategy="no", seed=args.seed,
        report_to="none", dataloader_drop_last=False,
    )
    CrossEncoderTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
    model.save_pretrained(str(out))
    (out / "fev_train_args.json").write_text(json.dumps(vars(args), indent=2))
    print(out)


if __name__ == "__main__":
    main()
