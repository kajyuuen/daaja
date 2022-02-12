from daaja.eda import RandamSwapAugmentor


def test_randam_swap_augmentor():
    augmentor = RandamSwapAugmentor(alpha=0.1)
    text = "日本語でデータ拡張を行う"
    new_text = augmentor.augment(text)
    assert len(new_text) == len(text)
    assert new_text != text
