# FinEvidence: evidence retrieval for numerical QA over full 10-K filings

Questions like *"what was the percentage change in operating lease expense from 2016 to 2017?"* need two things.
The first is finding the one table or paragraph in an ~86k-word annual report that holds the numbers.
The second is doing the arithmetic correctly. This project builds and measures both steps:

1. **Retrieval**: BM25 over table-aware chunks, then a **cross-encoder reranker fine-tuned on hard negatives**.
2. **Answering**: the LLM writes an **arithmetic expression that we execute** in a sandboxed AST evaluator.
   The LLM does not produce a free-text number.

Data: [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) (full 10-K text) joined with
[FinQA](https://github.com/czyssrs/FinQA) by question text, to borrow FinQA's expert-annotated evidence facts.
All tuning was done on dev. Test was run **once**, with the configuration frozen beforehand.

## Headline results (test, 920 questions / 240 filings / 78k chunks)

| retriever (rerank BM25 top-30) | params | hit@1 | hit@3 | hit@5 |
|---|---|---|---|---|
| BM25 | – | .504 | .711 | .784 |
| bge-small dense | 33M | .375 | .531 | .610 |
| bge-reranker-base, zero-shot | 278M | .571 | .777 | .840 |
| bge-reranker-v2-m3, zero-shot | 568M | .642 | .823 | .874 |
| **bge-reranker-base, fine-tuned (ours)** | 278M | **.737** | **.855** | **.900** |

hit@k is computed over the 843 test questions that have a labeled gold chunk.

| answer context (gpt-4.1-mini, executed expression) | accuracy |
|---|---|
| expert gold facts (upper bound) | 76.9% |
| FinQA's single gold page (no retrieval needed) | 72.6% |
| same page, direct answer instead of an expression | 65.7% |
| labeled gold chunk of the 10-K (oracle retrieval) | 60.7% |
| BM25 top-5 | 48.8% |
| BM25 → v2-m3, top-3 | 50.2% |
| **BM25 → fine-tuned reranker, top-3 (ours)** | **53.3%** |

Paired bootstrap, 10k resamples, 95% CI:

| comparison | Δ accuracy | 95% CI |
|---|---|---|
| executed expression vs direct answer (page context) | +7.0 | [+4.2, +9.7] |
| ours vs BM25 top-5 | +4.5 | [+2.0, +6.8] |
| ours vs zero-shot v2-m3 reranker | +3.0 | [+0.8, +5.3] |

**Dev, for honesty.** On dev (776 questions) the fine-tuned reranker gave the same retrieval gain
(hit@1 .728 vs .623 for v2-m3). The end-to-end gain over v2-m3 was not significant, though: +0.1, CI [−2.2, +2.4].
The retrieval improvement is robust. The end-to-end improvement over a strong zero-shot reranker is real on test
but small, and it varies across splits. Full tables are in `results/report_dev.md` and `results/report_test.md`.

## Where the remaining errors are (test, ours)

| bucket | questions | accuracy |
|---|---|---|
| gold chunk in the top 3 | 721 | 66.0% |
| gold chunk labeled but missed | 122 | 6.6% |
| no gold chunk could be labeled | 77 | 7.8% |

When retrieval finds the chunk, accuracy (66.0%) matches the oracle-chunk ceiling (66.0% on labeled questions).
So **retrieval is no longer the main bottleneck**. The gap to the 76.9% expert-fact ceiling comes from the filings themselves:
DocFinQA's raw 10-K text often states numbers differently from FinQA's curated facts (other units, restated values,
tables split during text extraction). The remaining errors are reasoning errors that also occur with perfect evidence.
Two prompt-side fixes made no difference on dev, so they were dropped:
reasoning-first prompting (49.0% vs 49.0%) and a separate evidence-extraction call (48.6–49.4%).

## Method

**Chunking** (`src/fev/chunk.py`). ~300-word chunks along paragraph boundaries. Markdown tables stay whole.
A table over 30 rows is split with its header repeated, so every piece keeps its years and units.
This gives 335k chunks across the three splits.

**Gold-chunk labels**. FinQA's evidence facts are mapped to 10-K chunks by numeric overlap. The label goes to the chunk(s)
covering the most gold numbers, provided they cover at least 50%. Years (1990–2030) are excluded, because they match
everywhere and otherwise produced ~5 tied chunks per question. 91% of questions get a label, with a median of 1 chunk.

**Reranker training** (`scripts/build_ce_data.py`, `scripts/train_reranker.py`). Trained on train-split questions only:
- up to 2 positives, 7 hard negatives from BM25's top 30, and 1 random negative per question;
- 48.6k pairs, 6.6k of them positive;
- hard negatives that contain ≥50% of the gold numbers are dropped as likely false negatives (3.6k skipped);
- `bge-reranker-base`, binary cross-entropy with `pos_weight`, 1 epoch, lr 2e-5, batch 16, max length 512.

Training took 70 minutes on an Apple-silicon laptop (MPS).

**Answering** (`src/fev/reason.py`). The top-3 chunks go to the prompt in document order. gpt-4.1-mini (temperature 0, JSON mode)
returns the evidence lines and one expression. `safe_eval` allows only number literals, `+ - * / **`
(exponent ≤ 10) and comparisons. An answer is correct within 1% relative error of FinQA's executed answer.
A lenient "×100" score is reported separately and never used for decisions.

## Reproduce

```bash
uv sync
uv run scripts/prepare.py                       # download DocFinQA + FinQA, chunk, label
uv run scripts/eval_retrieval.py --split train --methods bm25
uv run scripts/build_ce_data.py --split train --first bm25
uv run scripts/train_reranker.py --data train.ce.bm25 --out reranker-base-ft
uv run scripts/eval_retrieval.py --split test --methods bm25 bm25+models/reranker-base-ft bm25+BAAI/bge-reranker-v2-m3
uv run scripts/qa.py --split test --contexts bm25+reranker-base-ft@30 --top 3   # needs OPENAI_API_KEY in .env
uv run scripts/report.py --split test
```

The full test QA run costs about $1 with gpt-4.1-mini. Per-question outputs are cached in `results/qa/`.

## Limitations

- Gold-chunk labels are automatic: numeric overlap, not human judgment. 9% of questions have no label and count as misses in hit@k.
- There is one answer model (gpt-4.1-mini) and one training seed. The end-to-end gain over v2-m3 was significant on test but not on dev.
- FinQA questions are single-filing. Retrieval is within one known 10-K, not across a corpus of filings.
