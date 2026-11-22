"""Answer generation: direct answers, or an arithmetic expression that we execute ourselves."""

from __future__ import annotations

import ast
import json
import operator
import os
import re

PROGRAM_PROMPT = """You answer questions about a company's financial report using only the context.

Context:
{context}

Question: {question}

Write ONE arithmetic expression that computes the answer, using numbers copied from the context
(no thousands separators, no units). Allowed: + - * / ** parentheses, and comparisons (> <) for yes/no
questions. For percentages, return the ratio itself (e.g. growth rate = (new - old) / old), not times 100.
Reply as JSON: {{"evidence": "<the context lines you used>", "expression": "<expression>"}}"""

COT_PROMPT = """You answer questions about a company's financial report using only the context.

Context:
{context}

Question: {question}

First reason briefly: which line or table cell holds each number you need (check the year, the row label,
and the units, e.g. thousands vs millions), and which operation the question asks for. Then write ONE
arithmetic expression using those numbers (no thousands separators, no units). Allowed: + - * / **
parentheses, and comparisons (> <) for yes/no questions. For percentages, return the ratio itself
(e.g. growth rate = (new - old) / old), not times 100. If the numbers come in different units, convert
them inside the expression.
Reply as JSON: {{"reasoning": "<short reasoning>", "expression": "<expression>"}}"""

DIRECT_PROMPT = """You answer questions about a company's financial report using only the context.

Context:
{context}

Question: {question}

Give the final numeric answer (or yes/no). For percentages, give the ratio (e.g. 0.153 for 15.3%).
Reply as JSON: {{"answer": "<number or yes/no>"}}"""

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
        ast.Gt: operator.gt, ast.Lt: operator.lt, ast.GtE: operator.ge, ast.LtE: operator.le}


def safe_eval(expr: str):
    """Evaluate +-*/** and comparisons over number literals; anything else raises ValueError."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _OPS:
            return _OPS[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        raise ValueError(f"unsupported: {ast.dump(n)[:60]}")

    expr = expr.replace(",", "").replace("$", "").replace("%", "").strip()
    return ev(ast.parse(expr, mode="eval"))


def parse_number(s):
    if isinstance(s, bool):
        return "yes" if s else "no"
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    if s in ("yes", "no"):
        return s
    m = re.search(r"-?\d[\d,]*\.?\d*", s)
    if not m:
        return None
    v = float(m.group().replace(",", ""))
    return v / 100 if s.endswith("%") else v


def is_correct(pred, gold, rel_tol: float = 1e-2, lenient_scale: bool = False) -> bool:
    """Numbers match within rel_tol (FinQA gold answers are rounded). lenient_scale also accepts x100 / x0.01."""
    gold = parse_number(gold)
    if pred is None or gold is None:
        return False
    if isinstance(gold, str) or isinstance(pred, str):
        return pred == gold
    cands = [pred, pred * 100, pred / 100] if lenient_scale else [pred]
    return any(abs(p - gold) <= rel_tol * max(abs(gold), 1e-9) + 1e-9 for p in cands)


class LLM:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        from fev import load_dotenv

        load_dotenv()
        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    def json_call(self, prompt: str) -> tuple[dict, dict]:
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0,
            response_format={"type": "json_object"}, max_tokens=900,
        )
        usage = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
        try:
            return json.loads(resp.choices[0].message.content), usage
        except json.JSONDecodeError:
            return {}, usage


def run_program(llm: LLM, context: str, question: str, prompt: str = PROGRAM_PROMPT) -> dict:
    out, usage = llm.json_call(prompt.format(context=context, question=question))
    expr = str(out.get("expression", ""))
    try:
        pred = parse_number(safe_eval(expr))
        err = None
    except Exception as e:  # noqa: BLE001 - any parse/eval failure is a wrong answer
        pred, err = None, f"{type(e).__name__}: {e}"
    return {"expression": expr, "evidence": out.get("evidence", out.get("reasoning", "")), "pred": pred, "error": err, "usage": usage}


def run_cot(llm: LLM, context: str, question: str) -> dict:
    return run_program(llm, context, question, prompt=COT_PROMPT)


def run_direct(llm: LLM, context: str, question: str) -> dict:
    out, usage = llm.json_call(DIRECT_PROMPT.format(context=context, question=question))
    return {"raw": out.get("answer"), "pred": parse_number(out.get("answer")), "usage": usage}
