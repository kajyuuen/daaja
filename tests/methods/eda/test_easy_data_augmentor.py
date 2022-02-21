from daaja.methods.eda.easy_data_augmentor import EasyDataAugmentor


def test_easy_data_augmentor():
    NUM_AUG = 9
    augmentor = EasyDataAugmentor(alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=NUM_AUG)
    text = "日本語でデータ拡張を行う"
    aug_texts = augmentor.augments(text)
    assert NUM_AUG + 1 == len(aug_texts)
