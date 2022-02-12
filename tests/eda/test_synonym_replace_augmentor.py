from daaja.eda import SynonymReplaceAugmentor


def test_synonym_replace_augmentor():
    augmentor = SynonymReplaceAugmentor(alpha=0.1)
    text = "日本語でデータ拡張を行う"
    new_text = augmentor.augment(text)
    assert new_text != text
