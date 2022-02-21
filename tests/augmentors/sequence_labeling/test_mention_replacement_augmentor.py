from daaja.augmentors.sequence_labeling.mention_replacement_augmentor import \
    MentionReplacementAugmentor
from daaja.augmentors.sequence_labeling.utils import get_entity_dict


def test_mention_replacement_augmentor():
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
    entity_dict = get_entity_dict(tokens_list, labels_list)
    augmentor = MentionReplacementAugmentor(entity_dict, p=1)
    target_tokens = ["君", "は", "隆弘", "君", "かい"]
    target_labels = ["O", "O", "B-PER", "O", "O"]

    new_tokens, _ = augmentor.augment(target_tokens, target_labels)
    assert new_tokens != target_tokens


def test_mention_replacement_augmentor_2():
    augmentor = MentionReplacementAugmentor({"PER": ["田中 太郎"]}, p=1)
    target_tokens = ["君", "は", "田中", "君", "かい"]
    target_labels = ["O", "O", "B-PER", "O", "O"]

    new_tokens, new_labels = augmentor.augment(target_tokens, target_labels)
    assert ["君", "は", "田中", "太郎", "君", "かい"] == new_tokens
    assert ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'] == new_labels
