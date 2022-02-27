from daaja.augmentors.sentence.back_translation_augmentor import \
    BackTranslationAugmentor


def test_back_translation_augmentor():
    augmentor = BackTranslationAugmentor()
    text = "日本語でデータ拡張を行う"
    new_text = augmentor.augment(text)
    assert new_text != text
