## dev: 776 questions, 706 with a labeled gold chunk

| retriever | hit@1 | hit@3 | hit@5 | hit@10 | hit@30 |
|---|---|---|---|---|---|
| bm25 | 0.518 | 0.697 | 0.778 | 0.867 | 0.953 |
| dense | 0.334 | 0.508 | 0.599 | 0.703 | 0.847 |
| rrf | 0.458 | 0.659 | 0.737 | 0.836 | 0.943 |
| bm25+bge-reranker-base@30 | 0.550 | 0.763 | 0.843 | 0.899 | 0.953 |
| bm25+bge-reranker-v2-m3@30 | 0.623 | 0.814 | 0.871 | 0.926 | 0.953 |
| bm25+reranker-base-ft@30 | 0.728 | 0.847 | 0.877 | 0.928 | 0.953 |

| context | accuracy | lenient (x100) | exec fail | input tokens |
|---|---|---|---|---|
| gold | 0.744 | 0.762 | 0.026 | 222 |
| page | 0.701 | 0.718 | 0.032 | 1037 |
| oracle | 0.590 | 0.604 | 0.076 | 612 |
| bm25.top5 | 0.464 | 0.476 | 0.081 | 1999 |
| bm25+bge-reranker-v2-m3@30.top3 | 0.490 | 0.501 | 0.070 | 1306 |
| bm25+reranker-base-ft@30.top3 | 0.491 | 0.503 | 0.057 | 1475 |

| comparison (B - A) | delta | 95% CI |
|---|---|---|
| bm25+reranker-base-ft@30.top3 vs bm25.top5 | +0.027 | [+0.003, +0.052] |
| bm25+reranker-base-ft@30.top3 vs bm25+bge-reranker-v2-m3@30.top3 | +0.001 | [-0.022, +0.024] |

Error decomposition (bm25+reranker-base-ft@30.top3):
- gold chunk in top 3: n=598, accuracy 0.609
- labeled, missed: n=108, accuracy 0.102
- unlabeled: n=70, accuracy 0.086
- oracle chunk accuracy on labeled questions: 0.647
