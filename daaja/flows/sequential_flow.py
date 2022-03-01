import random
from typing import List

from daaja.augmentors.sentence.sentence_augmentor import SentenceAugmentor
from tqdm import tqdm


class SequentialSentenceFlow:
    def __init__(self,
                 augmentors: List[SentenceAugmentor],
                 num_aug: int,
                 verbose: bool = True) -> None:
        self.augmentors = augmentors
        self.num_aug = num_aug
        self.verbose = verbose

    def augments(self, sentence: str) -> List[str]:
        augmented_sentences = []
        num_per_technique = int(self.num_aug / len(self.augmentors)) + 1

        for augmentor in tqdm(self.augmentors, desc="augment", disable=not self.verbose):
            for _ in range(num_per_technique):
                augmented_sentences.append(augmentor.augment(sentence))

        random.shuffle(augmented_sentences)

        if self.num_aug >= 1:
            augmented_sentences = augmented_sentences[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        # Append the original sentence
        augmented_sentences.append(sentence)
        return augmented_sentences
