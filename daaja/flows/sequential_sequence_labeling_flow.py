import random
from typing import List, Tuple

from daaja.augmentors.sequence_labeling.sequence_labeling_augmentor import \
    SequenceLabelingAugmentor
from tqdm import tqdm


class SequentialSequenceLabelingFlow:
    def __init__(self,
                 augmentors: List[SequenceLabelingAugmentor],
                 num_aug: int,
                 verbose: bool = True) -> None:
        self.augmentors = augmentors
        self.num_aug = num_aug
        self.verbose = verbose

    def augments(self, tokens: List[str], labels: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        augmented_tokens_and_labels = []
        num_per_technique = int(self.num_aug / len(self.augmentors)) + 1

        for augmentor in tqdm(self.augmentors, desc="augment", disable=not self.verbose):
            for _ in range(num_per_technique):
                aug_tokens, aug_labels = augmentor.augment(tokens, labels)
                augmented_tokens_and_labels.append([aug_tokens, aug_labels])

        random.shuffle(augmented_tokens_and_labels)

        if self.num_aug >= 1:
            augmented_tokens_and_labels = augmented_tokens_and_labels[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_tokens_and_labels)
            augmented_tokens_and_labels = [s for s in augmented_tokens_and_labels if random.uniform(0, 1) < keep_prob]

        # Append the original sentence
        augmented_tokens_and_labels.append([tokens, labels])
        augmented_tokens_list = [tokens_list for tokens_list, _ in augmented_tokens_and_labels]
        augmented_labels_list = [labels_list for _, labels_list in augmented_tokens_and_labels]
        return augmented_tokens_list, augmented_labels_list
