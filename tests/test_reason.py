import pytest

from fev.chunk import chunk_document, label_gold_chunks, numbers
from fev.reason import is_correct, parse_number, safe_eval


def test_safe_eval_arithmetic_and_compare():
    assert safe_eval("(1,200 - 1000) / 1000") == pytest.approx(0.2)
    assert safe_eval("637 / 5.0") == pytest.approx(127.4)
    assert safe_eval("3 > 2") is True


def test_safe_eval_rejects_code():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('ls')")
    with pytest.raises(ValueError):
        safe_eval("2 ** 1000")


def test_parse_and_correct():
    assert parse_number("15.3%") == pytest.approx(0.153)
    assert parse_number(True) == "yes"
    assert is_correct(127.41, 127.4)
    assert not is_correct(15.3, 0.153)
    assert is_correct(15.3, 0.153, lenient_scale=True)
    assert is_correct("yes", "yes")


def test_chunk_keeps_table_header_and_labels_gold():
    rows = "\n".join(f"| item {i} | {1000 + i} |" for i in range(70))
    doc = "intro text\n\n| name | value |\n| :--- | :--- |\n" + rows + "\n\nclosing text"
    chunks = chunk_document(doc, max_words=50, table_rows=30)
    table_chunks = [c for c in chunks if "| name | value |" in c]
    assert len(table_chunks) == 3
    assert numbers("$2,457 and 50.3") == {"2457", "50.3"}
    gold = label_gold_chunks(chunks, {"table_5": "the item 40 is 1040 ; item 41 is 1041"})
    assert gold and all("1040" in chunks[i] for i in gold)
