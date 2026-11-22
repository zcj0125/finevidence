"""Split a full filing into retrieval chunks, and label which chunks hold an example's gold evidence.

Paragraphs are packed into ~max_words chunks. A markdown table is never split across a paragraph boundary;
a long table is cut every `table_rows` rows and each piece repeats the table's header lines.
Gold labels: FinQA's evidence is in its own linearization, so a chunk is labeled by how many of the gold
facts' numbers it contains (numbers normalized: no $ or thousands separators).
"""

from __future__ import annotations

import re

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize(text: str) -> str:
    return text.replace("\\t", " ").replace("\t", " ")


def _table_pieces(block: str, table_rows: int) -> list[str]:
    lines = block.splitlines()
    sep = next((i for i, l in enumerate(lines) if set(l.replace("|", "").strip()) <= set(":- ") and "-" in l), None)
    head_n = 2 if sep is None else sep + 1
    head, body = lines[:head_n], lines[head_n:]
    if len(body) <= table_rows:
        return [block]
    return ["\n".join(head + body[i:i + table_rows]) for i in range(0, len(body), table_rows)]


def chunk_document(text: str, max_words: int = 300, table_rows: int = 30) -> list[str]:
    blocks = []
    for para in re.split(r"\n\s*\n", normalize(text)):
        para = para.strip()
        if not para:
            continue
        blocks += _table_pieces(para, table_rows) if para.lstrip().startswith("|") else [para]
    chunks, cur, n = [], [], 0
    for b in blocks:
        w = len(b.split())
        if cur and n + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(b)
        n += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def numbers(text: str) -> set[str]:
    out = set()
    for m in NUM.findall(text.replace("$", " ")):
        s = m.replace(",", "").rstrip(".")
        if s.lstrip("-") in {"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            continue
        try:
            out.add(f"{float(s):g}")
        except ValueError:
            pass
    return out


def label_gold_chunks(chunks: list[str], gold_inds: dict[str, str], min_frac: float = 0.5) -> list[int]:
    """Chunks covering the most gold numbers (at least min_frac of them). Empty if evidence has no numbers."""
    gold = numbers(" ".join(gold_inds.values()))
    if not gold:
        return []
    cover = [len(gold & numbers(c)) / len(gold) for c in chunks]
    best = max(cover)
    if best < min_frac:
        return []
    return [i for i, c in enumerate(cover) if c == best]
