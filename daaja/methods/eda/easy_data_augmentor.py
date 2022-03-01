from typing import List

from daaja.augmentors.sentence.randam_delete_augmentor import \
    RandamDeleteAugmentor
from daaja.augmentors.sentence.randam_insert_augmentor import \
    RandamInsertAugmentor
from daaja.augmentors.sentence.randam_swap_augmentor import RandamSwapAugmentor
from daaja.augmentors.sentence.synonym_replace_augmentor import \
    SynonymReplaceAugmentor
from daaja.flows.sequential_flow import SequentialSentenceFlow
from daaja.resouces import Resouces
from daaja.tokenizer import Tokenizer


class EasyDataAugmentor:
    def __init__(self, alpha_sr: float, alpha_ri: float, alpha_rs: float, p_rd: float, num_aug: int, verbose: bool = True):
        resouces = Resouces()
        self.tokenizer = Tokenizer()

        self.flow = SequentialSentenceFlow(
            augmentors=[
                SynonymReplaceAugmentor(alpha=alpha_sr, resouces=resouces, tokenizer=self.tokenizer),
                RandamInsertAugmentor(alpha=alpha_ri, resouces=resouces, tokenizer=self.tokenizer),
                RandamSwapAugmentor(alpha=alpha_rs, tokenizer=self.tokenizer),
                RandamDeleteAugmentor(p=p_rd, tokenizer=self.tokenizer)],
            num_aug=num_aug,
            verbose=verbose)

    def augments(self, sentence: str) -> List[str]:
        return self.flow.augments(sentence)
