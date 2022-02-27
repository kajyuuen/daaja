from daaja.augmentors.sentence.contextual_augmentor import ContextualAugmentor


def test_contextual_augmentor():
    augmentor = ContextualAugmentor()
    text = "日本語でデータ拡張を行う"
    new_text = augmentor.augment(text)
    assert new_text != text
