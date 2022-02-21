from daaja.methods.ner_sda.simple_data_augmentation_for_ner import \
    SimpleDataAugmentationforNER


def test_simple_data_augmentation_for_ner():
    NUM_AUG = 4
    tokens_list = [
        ["私", "は", "田中", "と", "いい", "ます"],
        ["筑波", "大学", "に", "所属", "して", "ます"],
        ["今日", "から", "筑波", "大学", "に", "通う"],
        ["茨城", "大学"],
    ]
    labels_list = [
        ["O", "O", "B-PER", "O", "O", "O"],
        ["B-ORG", "I-ORG", "O", "O", "O", "O"],
        ["B-DATE", "O", "B-ORG", "I-ORG", "O", "O"],
        ["B-ORG", "I-ORG"],
    ]
    augmentor = SimpleDataAugmentationforNER(tokens_list=tokens_list, labels_list=labels_list,
                                             p_power=1, p_lwtr=0.3, p_mr=0.3, p_sis=0.3, p_sr=0.3, num_aug=NUM_AUG)
    tokens = ["吉田", "さん", "は", "株式", "会社", "A", "に", "出張", "予定", "だ"]
    labels = ["B-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O"]
    augmented_tokens_list, augmented_labels_list = augmentor.augments(tokens, labels)
    assert NUM_AUG + 1 == len(augmented_tokens_list)
    assert NUM_AUG + 1 == len(augmented_labels_list)
