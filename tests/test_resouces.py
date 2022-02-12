from daaja.resouces import Resouces


def test_get_synonyms():
    wordnet = Resouces()
    synonyms = wordnet.get_synonyms("データ")
    assert 6 == len(synonyms)
    assert "情報" in wordnet.get_synonyms("データ")


def test_get_stopwords():
    wordnet = Resouces()
    stopwords = wordnet.stopwords
    assert 0 < len(stopwords)
