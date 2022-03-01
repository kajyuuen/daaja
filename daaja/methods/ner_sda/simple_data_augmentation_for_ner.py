from typing import List, Tuple

from daaja.augmentors.sequence_labeling.labelwise_token_replacement_augmentor import \
    LabelwiseTokenReplacementAugmentor
from daaja.augmentors.sequence_labeling.mention_replacement_augmentor import \
    MentionReplacementAugmentor
from daaja.augmentors.sequence_labeling.shuffle_within_segments_augmentor import \
    ShuffleWithinSegmentsAugmentor
from daaja.augmentors.sequence_labeling.synonym_replacement_augmentor import \
    SynonymReplacementAugmentor
from daaja.augmentors.sequence_labeling.utils import (get_entity_dict,
                                                      get_token2prob_in_label)
from daaja.flows.sequential_sequence_labeling_flow import \
    SequentialSequenceLabelingFlow
from daaja.resouces import Resouces


class SimpleDataAugmentationforNER:
    def __init__(self,
                 tokens_list: List[List[str]],
                 labels_list: List[List[str]],
                 p_power: float,
                 p_lwtr: float,
                 p_mr: float,
                 p_sis: float,
                 p_sr: float,
                 num_aug: int,
                 verbose: bool = True):
        resouces = Resouces()
        entity_dict = get_entity_dict(tokens_list, labels_list)
        token_and_prob_in_label = get_token2prob_in_label(tokens_list, labels_list, p_power=p_power)

        self.flow = SequentialSequenceLabelingFlow(
            augmentors=[
                LabelwiseTokenReplacementAugmentor(token_and_prob_in_label, p=p_lwtr, resouces=resouces),
                MentionReplacementAugmentor(entity_dict=entity_dict, p=p_mr),
                ShuffleWithinSegmentsAugmentor(p=p_sis),
                SynonymReplacementAugmentor(p=p_sr, resouces=resouces)
            ],
            num_aug=num_aug,
            verbose=verbose
        )

    def augments(self, tokens: List[str], labels: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        return self.flow.augments(tokens, labels)
