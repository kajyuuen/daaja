import random

from daaja.augmentors.sentence.sentence_augmentor import SentenceAugmentor
from daaja.tokenizer import Tokenizer


class RandamDeleteAugmentor(SentenceAugmentor):
    def __init__(self,
                 p: float = 0.05,
                 tokenizer: Tokenizer = Tokenizer()) -> None:
        self.p = p
        self.tokenizer = tokenizer

    def augment(self, sentence: str) -> str:
        tokens, _ = self.tokenizer.tokenize(sentence)
        if len(tokens) == 1:
            return sentence

        converted_tokens = []
        for token in tokens:
            r = random.uniform(0, 1)
            if r < self.p:
                continue
            converted_tokens.append(token)

        # None of the tokens exist
        if len(converted_tokens) == 0:
            random_idx = random.randint(0, len(tokens) - 1)
            return "".join(tokens[random_idx])

        return "".join(converted_tokens)
