import random

import torch
from daaja.augmentors.sentence.sentence_augmentor import SentenceAugmentor
from transformers import pipeline


class ContextualAugmentor(SentenceAugmentor):
    def __init__(self,
                 p: float = 0.15,
                 max_predictions_per_seq: int = 100000,
                 model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking",
                 device: int = -1) -> None:
        self.p = p
        self.max_predictions_per_seq = max_predictions_per_seq
        self.pipeline = pipeline("fill-mask",
                                 model=model_name,
                                 top_k=2,
                                 device=device)
        self.mask_token = self.pipeline.tokenizer.mask_token

    @torch.no_grad()
    def augment(self, sentence: str) -> str:
        """Contextual Augmentor

        Args:
            sentence (str): sentence

        Returns:
            str: Contextual augmented sentence
        """
        tokens = self.pipeline.tokenizer.tokenize(sentence)
        n = min(self.max_predictions_per_seq, max(1, int(self.p * len(tokens))))
        random_idx = list(range(len(tokens)))
        random.shuffle(random_idx)

        converted_tokens = tokens.copy()
        for random_i in random_idx[:n]:
            converted_tokens[random_i] = self.mask_token
            aug_texts = self.pipeline("".join(converted_tokens))
            for aug_text in aug_texts:
                token_str = aug_text["token_str"].replace(" ", "")
                if token_str == tokens[random_i]:
                    continue
                converted_tokens[random_i] = token_str
                break

        return "".join(converted_tokens)
