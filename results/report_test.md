## test: 920 questions, 843 with a labeled gold chunk

| retriever | hit@1 | hit@3 | hit@5 | hit@10 | hit@30 |
|---|---|---|---|---|---|
| bm25 | 0.504 | 0.711 | 0.784 | 0.853 | 0.948 |
| dense | 0.375 | 0.531 | 0.610 | 0.718 | 0.859 |
| rrf | 0.460 | 0.661 | 0.736 | 0.829 | 0.931 |
| bm25+bge-reranker-base@30 | 0.571 | 0.777 | 0.840 | 0.905 | 0.948 |
| bm25+bge-reranker-v2-m3@30 | 0.642 | 0.823 | 0.874 | 0.912 | 0.948 |
| bm25+reranker-base-ft@30 | 0.737 | 0.855 | 0.900 | 0.930 | 0.948 |

| context | accuracy | lenient (x100) | exec fail | input tokens |
|---|---|---|---|---|
| gold | 0.768 | 0.798 | 0.020 | 221 |
| page | 0.726 | 0.754 | 0.025 | 1015 |
| page.direct | 0.657 | 0.673 | nan | 968 |
| oracle | 0.607 | 0.623 | 0.074 | 592 |
| bm25.top5 | 0.488 | 0.509 | 0.072 | 2004 |
| bm25+bge-reranker-v2-m3@30.top3 | 0.502 | 0.525 | 0.082 | 1309 |
| bm25+reranker-base-ft@30.top3 | 0.533 | 0.550 | 0.061 | 1468 |

| comparison (B - A) | delta | 95% CI |
|---|---|---|
| page.direct vs page | -0.070 | [-0.097, -0.042] |
| bm25+reranker-base-ft@30.top3 vs bm25.top5 | +0.045 | [+0.020, +0.068] |
| bm25+reranker-base-ft@30.top3 vs bm25+bge-reranker-v2-m3@30.top3 | +0.030 | [+0.008, +0.053] |

Error decomposition (bm25+reranker-base-ft@30.top3):
- gold chunk in top 3: n=721, accuracy 0.660
- labeled, missed: n=122, accuracy 0.066
- unlabeled: n=77, accuracy 0.078
- oracle chunk accuracy on labeled questions: 0.660
