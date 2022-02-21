from daaja.augmentors.ner.shuffle_within_segments_augmentor import \
    ShuffleWithinSegmentsAugmentor


def test_shuffle_within_segments_augmentor():
    augmentor = ShuffleWithinSegmentsAugmentor(p=1)
    target_tokens = ["君", "は", "筑波", "大学", "出身", "の", "田中", "君", "かい"]
    target_labels = ["O", "O", "B-ORG", "B-ORG", "O", "O", "B-PER", "O", "O"]

    new_tokens, new_labels = augmentor.augment(target_tokens, target_labels)
    assert sorted(new_tokens) == sorted(target_tokens)
    assert sorted(new_labels) == sorted(target_labels)
