from pathlib import Path

from daaja.methods.ner_sda.run import load_conll


def test_load_conll():
    tsv_path = Path("./tests/methods/sda_ner/fixtures/text.tsv")
    results = load_conll(tsv_path)
    assert [['私', 'は', '田中', 'と', 'いい', 'ます'], ['私', 'は', '小林', 'と', 'いい', 'ます']] == results[0]
    assert [['O', 'O', 'B-PER', 'O', 'O', 'O'], ['O', 'O', 'B-PER', 'O', 'O', 'O']] == results[1]
