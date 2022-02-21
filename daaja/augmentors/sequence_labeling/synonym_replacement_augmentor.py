import random
from typing import List, Tuple

import numpy as np
from daaja.augmentors.sequence_labeling.sequence_labeling_augmentor import \
    SequenceLabelingAugmentor
from daaja.resouces import Resouces


class SynonymReplacementAugmentor(SequenceLabelingAugmentor):
    def __init__(self,
                 p: float = 0.1,
                 resouces: Resouces = Resouces()) -> None:
        self.p = p
        self.resouces = resouces

    def augment(self,
                tokens: List[str],
                labels: List[str]) -> Tuple[List[str], List[str]]:
        masks = np.random.binomial(1, self.p, len(tokens))
        generated_tokens = []
        for mask, token, label in zip(masks, tokens, labels):
            if mask == 0 or self.resouces.is_stopword(token):
                generated_token = token
            else:
                synonyms_set = set(self.resouces.get_synonyms(token))
                if token in synonyms_set:
                    synonyms_set.remove(token)
                if len(synonyms_set) == 0:
                    generated_token = token
                else:
                    synonym = random.choice(list(synonyms_set))
                    generated_token = synonym
            generated_tokens.append(generated_token)

        return generated_tokens, labels
