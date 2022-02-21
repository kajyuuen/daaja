import random
from typing import List

from daaja.augmentors.sentence.sentence_augmentor import SentenceAugmentor
from daaja.tokenizer import Tokenizer


class RandamSwapAugmentor(SentenceAugmentor):
    def __init__(self,
                 alpha: float = 0.1,
                 tokenizer: Tokenizer = Tokenizer()) -> None:
        self.alpha = alpha
        self.tokenizer = tokenizer

    def augment(self, sentence: str) -> str:
        """Randomly swap two words in the sentence n times

        Args:
            sentence (str): sentence

        Returns:
            str: swaped sentence
        """
        tokens, _ = self.tokenizer.tokenize(sentence)
        n = max(1, int(self.alpha * len(tokens)))

        converted_tokens = tokens.copy()
        for _ in range(n):
            converted_tokens = self._swap_word(converted_tokens)
        return "".join(converted_tokens)

    def _swap_word(self, tokens: List[str]) -> List[str]:
        random_idx_1 = random.randint(0, len(tokens) - 1)
        random_idx_2 = random_idx_1
        cnt = 0
        while random_idx_1 == random_idx_2:
            random_idx_2 = random.randint(0, len(tokens) - 1)
            cnt += 1
            if cnt > 3:
                return tokens
        tokens[random_idx_1], tokens[random_idx_2] = tokens[random_idx_2], tokens[random_idx_1]
        return tokens
