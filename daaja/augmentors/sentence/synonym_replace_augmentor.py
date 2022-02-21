import random

from daaja.augmentors.sentence.sentence_augmentor import SentenceAugmentor
from daaja.resouces import Resouces
from daaja.tokenizer import Tokenizer


class SynonymReplaceAugmentor(SentenceAugmentor):
    def __init__(self,
                 alpha: float = 0.1,
                 resouces: Resouces = Resouces(),
                 tokenizer: Tokenizer = Tokenizer()) -> None:
        self.alpha = alpha
        self.resouces = resouces
        self.tokenizer = tokenizer

    def augment(self, sentence: str) -> str:
        """Randomly replace n tokens with their synonyms.

        Args:
            sentence (str): sentence

        Returns:
            str: synonym_replaced_sentence
        """
        tokens, selected_tokens = self.tokenizer.tokenize(sentence)
        n = max(1, int(self.alpha * len(tokens)))

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
        return "".join(converted_tokens)
