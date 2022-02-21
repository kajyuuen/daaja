from daaja.augmentors.ner.utils import get_entity_dict, get_token2prob_in_label


def test_get_token2prob_in_label():
    tokens_list = [
        ["私", "は", "田中", "と", "いいます"],
        ["筑波", "大学", "に", "所属", "してます"],
        ["今日", "から", "筑波", "大学", "に", "通う"],
        ["茨城", "大学"],
    ]
    labels_list = [
        ["O", "O", "B-PER", "O", "O"],
        ["B-ORG", "I-ORG", "O", "O", "O"],
        ["B-DATE", "O", "B-ORG", "I-ORG", "O", "O"],
        ["B-ORG", "I-ORG"],
    ]
    result = get_token2prob_in_label(tokens_list, labels_list)
    assert "O" in result.keys()
    assert "B-PER" in result.keys()
    assert "B-ORG" in result.keys()
    assert "I-ORG" in result.keys()
    assert "B-DATE" in result.keys()
    assert (["今日"], [1.0]) == result["B-DATE"]


def test_get_entity_dict():
    tokens_list = [
        ["私", "は", "田中", "と", "いいます"],
        ["筑波", "大学", "に", "所属", "してます"],
        ["今日", "から", "筑波", "大学", "に", "通う"],
        ["茨城", "大学"],
    ]
    labels_list = [
        ["O", "O", "B-PER", "O", "O"],
        ["B-ORG", "I-ORG", "O", "O", "O"],
        ["B-DATE", "O", "B-ORG", "I-ORG", "O", "O"],
        ["B-ORG", "I-ORG"],
    ]
    result = get_entity_dict(tokens_list, labels_list)
    assert "O" in result.keys()
    assert "PER" in result.keys()
    assert "ORG" in result.keys()
    assert "DATE" in result.keys()
    assert "今日" in result["DATE"]
