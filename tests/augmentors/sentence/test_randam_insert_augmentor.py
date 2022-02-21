from daaja.augmentors.sentence.randam_insert_augmentor import \
    RandamInsertAugmentor


def test_randam_insert_augmentor():
    augmentor = RandamInsertAugmentor(alpha=0.1)
    text = "日本語でデータ拡張を行う"
    new_text = augmentor.augment(text)
    assert len(new_text) > len(text)
