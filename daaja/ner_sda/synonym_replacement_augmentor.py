import random
from typing import List, Tuple

import numpy as np
from daaja.ner_augmentor import NerAugmentor
from daaja.ner_sda.utils import TokenAndProbInLabel
from daaja.resouces import Resouces


class SynonymReplacementAugmentor(NerAugmentor):
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
                synonyms_set = set(self.resouces.get_synonyms(token))
                if token in synonyms_set:
                    synonyms_set.remove(token)
                if len(synonyms_set) == 0:
                    generated_token = token
                else:
                    # NOTE: トークナイザーによっては予期してないトークンが入る可能性がある
                    # 例: 秘密捜査員はトークナイザーによっては秘密/捜査員になる
                    synonym = random.choice(list(synonyms_set))
                    generated_token = synonym
            generated_tokens.append(generated_token)

        return generated_tokens, labels
