from daaja.augmentors.sequence_labeling.utils import get_token2prob_in_label
from daaja.augmentors.sequence_labeling.labelwise_token_replacement_augmentor import \
    LabelwiseTokenReplacementAugmentor


def test_labelwise_token_replacement_augmentor():
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
    token_and_prob_in_label = get_token2prob_in_label(tokens_list, labels_list)
    augmentor = LabelwiseTokenReplacementAugmentor(token_and_prob_in_label, p=1)
    target_tokens = ["君", "は", "田中", "君", "かい"]
    target_labels = ["O", "O", "B-PER", "O", "O"]

    new_tokens, new_labels = augmentor.augment(target_tokens, target_labels)
    assert new_tokens != target_tokens
    assert new_labels == target_labels
