from pathlib import Path

from daaja.eda.run import load_tsv


def test_load_tsv():
    tsv_path = Path("./tests/eda/fixtures/text.tsv")
    results = load_tsv(tsv_path)
    assert 5 == len(results)
