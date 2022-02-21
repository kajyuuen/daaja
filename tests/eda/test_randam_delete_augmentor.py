from daaja.augmentors.sentence.randam_delete_augmentor import \
    RandamDeleteAugmentor


def test_randam_delete_augmentor():
    augmentor = RandamDeleteAugmentor(p=1)
    text = "日本語でデータ拡張を行う"
    new_text = augmentor.augment(text)
    assert len(new_text) < len(text)
