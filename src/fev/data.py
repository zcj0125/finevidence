"""Load FinQA (one report page per question) and DocFinQA (the full 10-K for the same questions).

DocFinQA has no evidence labels, so each DocFinQA question is joined to its FinQA twin by question text
(ties broken by answer), which brings over the expert-annotated gold facts and program.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from fev import PROJECT_ROOT

DATA = PROJECT_ROOT / "data"


@dataclass
class Example:
    id: str
    question: str
    answer: float | str  # FinQA exe_ans: a number, or "yes"/"no"
    program: str
    page_text: list[str]  # pre_text + post_text sentences
    table: list[list[str]]  # first row is the header
    gold_inds: dict[str, str]  # e.g. {"table_3": "...", "text_12": "..."} in FinQA's own linearization
    filename: str
    doc_id: int | None = None  # index into the DocFinQA document list
    extra: dict = field(default_factory=dict)


def table_markdown(table: list[list[str]]) -> str:
    if not table:
        return ""
    head, *rows = table
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def page_context(ex: Example) -> str:
    n_pre = ex.extra.get("n_pre", len(ex.page_text))
    pre, post = ex.page_text[:n_pre], ex.page_text[n_pre:]
    return "\n".join(pre) + "\n\n" + table_markdown(ex.table) + "\n\n" + "\n".join(post)


def gold_context(ex: Example) -> str:
    return "\n".join(ex.gold_inds.values())


def load_finqa(split: str) -> list[Example]:
    out = []
    for e in json.load((DATA / "finqa" / f"{split}.json").open()):
        qa = e["qa"]
        out.append(Example(
            id=e["id"], question=qa["question"].strip(), answer=qa["exe_ans"], program=qa["program"],
            page_text=e["pre_text"] + e["post_text"], table=e["table"], gold_inds=qa["gold_inds"],
            filename=e["filename"], extra={"n_pre": len(e["pre_text"])},
        ))
    return out


def load_docfinqa(split: str = "dev") -> tuple[list[Example], list[str]]:
    """Return FinQA examples that have a DocFinQA twin (doc_id set) and the list of unique full documents."""
    finqa = {}
    for ex in load_finqa(split):
        finqa.setdefault(ex.question.lower(), []).append(ex)
    docs, doc_index, out = [], {}, []
    for row in json.load((DATA / "docfinqa" / f"{split}.json").open()):
        cands = finqa.get(row["Question"].strip().lower(), [])
        if len(cands) > 1:
            cands = [c for c in cands if _same_answer(c.answer, row["Answer"])] or cands[:1]
        if len(cands) != 1:
            continue
        key = row["Context"][:4000]
        if key not in doc_index:
            doc_index[key] = len(docs)
            docs.append(row["Context"])
        ex = cands[0]
        out.append(Example(**{**ex.__dict__, "doc_id": doc_index[key], "extra": dict(ex.extra)}))
    return out, docs


def _same_answer(a, b) -> bool:
    try:
        return abs(float(a) - float(str(b).replace(",", "").rstrip("%"))) < 1e-2 * max(1.0, abs(float(a)))
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()
