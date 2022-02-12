import random
from typing import List

from daaja.augmentor import Augmentor
from daaja.tokenizer import Tokenizer


class RandamDeleteAugmentor(Augmentor):
    def __init__(self,
                 p: float = 0.05,
                 tokenizer: Tokenizer = Tokenizer()) -> None:
        self.p = p
        self.tokenizer = tokenizer

    def augment(self, sentence: str) -> str:
        tokens, _ = self.tokenizer.tokenize(sentence)
        converted_tokens = self.random_delete(tokens, self.p)
        return "".join(converted_tokens)

    def random_delete(self, tokens: List[str], p: float) -> List[str]:
        """Randomly delete tokens from the sentence with probability p

        Args:
            tokens (List[str]): Sentence
            p (float): probability

        Returns:
            List[str]: tokens with n tokens removed
        """
        if len(tokens) == 1:
            return tokens

        converted_tokens = []
        for token in tokens:
            r = random.uniform(0, 1)
            if r < p:
                continue
            converted_tokens.append(token)

        if len(converted_tokens) == 0:
            random_idx = random.randint(0, len(tokens) - 1)
            return [tokens[random_idx]]

        return converted_tokens
