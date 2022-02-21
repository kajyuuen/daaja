import random
from typing import List, Optional

from daaja.augmentors.sentence.sentence_augmentor import SentenceAugmentor
from daaja.resouces import Resouces
from daaja.tokenizer import Tokenizer


class RandamInsertAugmentor(SentenceAugmentor):
    def __init__(self,
                 alpha: float = 0.1,
                 resouces: Resouces = Resouces(),
                 tokenizer: Tokenizer = Tokenizer()) -> None:
        self.alpha = alpha
        self.resouces = resouces
        self.tokenizer = tokenizer

    def augment(self, sentence: str) -> str:
        """Randomly insert a selected token synonym from selected_tokens in sentence.

        Args:
            sentence (str): sentence

        Returns:
            str: swaped sentence
        """
        tokens, selected_tokens = self.tokenizer.tokenize(sentence)
        n = max(1, int(self.alpha * len(tokens)))

        converted_tokens = tokens.copy()
        for _ in range(n):
            converted_tokens = self._add_word(converted_tokens, selected_tokens)
        return "".join(converted_tokens)

    def _add_word(self, tokens: List[str], selected_tokens: List[Optional[str]]) -> List[str]:
        synonyms = []
        selected_not_none_tokens: List[str] = [token for token in selected_tokens if token is not None]
        cnt = 0
        while len(synonyms) < 1:
            random_selected_token = selected_not_none_tokens[random.randint(0, len(selected_not_none_tokens) - 1)]
            synonyms = self.resouces.get_synonyms(random_selected_token)
            cnt += 1
            # NOTE: 10 is magic number.
            # If this function don't see a synonym after looking for it 10 times, done.
            if cnt >= 10:
                return tokens
        random_idx = random.randint(0, len(tokens) - 1)
        random_selected_synonym = synonyms[0]
        tokens.insert(random_idx, random_selected_synonym)
        return tokens
