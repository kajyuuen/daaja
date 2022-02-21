from daaja.augmentors.ner.synonym_replacement_augmentor import \
    SynonymReplacementAugmentor


def test_synonym_replacement_augmentor():
    augmentor = SynonymReplacementAugmentor(p=1)
    target_tokens = ["君", "は", "田中", "君", "かい"]
    target_labels = ["O", "O", "B-PER", "O", "O"]

    new_tokens, new_labels = augmentor.augment(target_tokens, target_labels)
    assert new_tokens != target_tokens
    assert new_labels == target_labels
