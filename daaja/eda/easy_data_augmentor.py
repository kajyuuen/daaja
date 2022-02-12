import random
from typing import List

from daaja.eda.randam_delete_augmentor import RandamDeleteAugmentor
from daaja.eda.randam_insert_augmentor import RandamInsertAugmentor
from daaja.eda.randam_swap_augmentor import RandamSwapAugmentor
from daaja.eda.synonym_replace_augmentor import SynonymReplaceAugmentor
from daaja.resouces import Resouces
from daaja.tokenizer import Tokenizer


class EasyDataAugmentor:
    def __init__(self, alpha_sr: float, alpha_ri: float, alpha_rs: float, p_rd: float, num_aug: int):
        resouces = Resouces()
        self.tokenizer = Tokenizer()

        # augmentors
        self.synonym_replace_augmentor = SynonymReplaceAugmentor(alpha=alpha_sr, resouces=resouces, tokenizer=self.tokenizer)
        self.randam_insert_augmentor = RandamInsertAugmentor(alpha=alpha_ri, resouces=resouces, tokenizer=self.tokenizer)
        self.randam_swap_augmentor = RandamSwapAugmentor(alpha=alpha_rs, tokenizer=self.tokenizer)
        self.randam_delete_augmentor = RandamDeleteAugmentor(p=p_rd, tokenizer=self.tokenizer)

        # params
        self.num_aug = num_aug

    def augments(self, sentence: str) -> List[str]:
        tokens, selected_tokens = self.tokenizer.tokenize(sentence)
        num_tokens = len(tokens)
        augmented_sentences = []
        num_new_per_technique = int(self.num_aug / 4) + 1

        # Synonym Replace
        if self.synonym_replace_augmentor.alpha > 0:
            n_sr = max(1, int(self.synonym_replace_augmentor.alpha * num_tokens))
            for _ in range(num_new_per_technique):
                aug_tokens = self.synonym_replace_augmentor.synonym_replace(tokens, selected_tokens, n_sr)
                augmented_sentences.append("".join(aug_tokens))

        # Random Insert
        if self.randam_insert_augmentor.alpha > 0:
            n_ri = max(1, int(self.randam_insert_augmentor.alpha * num_tokens))
            for _ in range(num_new_per_technique):
                aug_tokens = self.randam_insert_augmentor.random_insert(tokens, selected_tokens, n_ri)
                augmented_sentences.append("".join(aug_tokens))

        # Random Swap
        if self.randam_swap_augmentor.alpha > 0:
            n_ri = max(1, int(self.randam_swap_augmentor.alpha * num_tokens))
            for _ in range(num_new_per_technique):
                aug_tokens = self.randam_swap_augmentor.random_swap(tokens, n_ri)
                augmented_sentences.append("".join(aug_tokens))

        # Random Delete
        if self.randam_delete_augmentor.p > 0:
            for _ in range(num_new_per_technique):
                p = self.randam_delete_augmentor.p
                aug_tokens = self.randam_delete_augmentor.random_delete(tokens, p)
                augmented_sentences.append("".join(aug_tokens))

        random.shuffle(augmented_sentences)
        if self.num_aug >= 1:
            augmented_sentences = augmented_sentences[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        # Append the original sentence
        augmented_sentences.append(sentence)
        return augmented_sentences
