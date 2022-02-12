import random
from typing import List, Optional

from daaja.augmentor import Augmentor
from daaja.resouces import Resouces
from daaja.tokenizer import Tokenizer


class SynonymReplaceAugmentor(Augmentor):
    def __init__(self,
                 alpha: float = 0.1,
                 resouces: Resouces = Resouces(),
                 tokenizer: Tokenizer = Tokenizer()) -> None:
        self.alpha = alpha
        self.resouces = resouces
        self.tokenizer = tokenizer

    def augment(self, sentence: str) -> str:
        tokens, selected_tokens = self.tokenizer.tokenize(sentence)
        n = max(1, int(self.alpha * len(tokens)))
        converted_tokens = self.synonym_replace(tokens, selected_tokens, n)
        return "".join(converted_tokens)

    def synonym_replace(self, tokens: List[str], selected_tokens: List[Optional[str]], n: int) -> List[str]:
        """Randomly replace n tokens with their synonyms.

        Args:
            tokens (List[str]): Sentence
            selected_tokens (List[Optional[str]]): List of words to look up synonyms for.
            n (int): Number of words to be inserted.

        Returns:
            List[str]: [description]
        """
        converted_tokens = tokens.copy()
        selected_token_idxes = [i for i, token in enumerate(selected_tokens) if token is not None]
        random.shuffle(selected_token_idxes)
        cnt = 0
        for idx in selected_token_idxes:
            token = tokens[idx]
            selected_token = selected_tokens[idx]
            synonyms = self.resouces.get_synonyms(selected_token)
            if len(synonyms) == 0:
                continue
            synonym = random.choice(synonyms)
            converted_tokens = [synonym if converted_token == token else converted_token for converted_token in converted_tokens]
            cnt += 1
            if cnt >= n:
                break
        return converted_tokens
