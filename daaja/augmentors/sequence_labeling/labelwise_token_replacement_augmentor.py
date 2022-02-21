from typing import List, Tuple

import numpy as np
from daaja.augmentors.sequence_labeling.sequence_labeling_augmentor import \
    SequenceLabelingAugmentor
from daaja.augmentors.sequence_labeling.utils import TokenAndProbInLabel
from daaja.resouces import Resouces


class LabelwiseTokenReplacementAugmentor(SequenceLabelingAugmentor):
    def __init__(self,
                 token_and_prob_in_label: TokenAndProbInLabel,
                 p: float = 0.1,
                 resouces: Resouces = Resouces()) -> None:
        self.p = p
        self.resouces = resouces
        self.token_and_prob_in_label = token_and_prob_in_label

    def augment(self,
                tokens: List[str],
                labels: List[str]) -> Tuple[List[str], List[str]]:
        masks = np.random.binomial(1, self.p, len(tokens))
        generated_tokens = []
        for mask, token, label in zip(masks, tokens, labels):
            if mask == 0 or self.resouces.is_stopword(token):
                generated_token = token
            else:
                random_idx = np.random.choice(len(self.token_and_prob_in_label[label][1]), 1, p=self.token_and_prob_in_label[label][1])[0]
                generated_token = self.token_and_prob_in_label[label][0][random_idx]
            generated_tokens.append(generated_token)

        return generated_tokens, labels
