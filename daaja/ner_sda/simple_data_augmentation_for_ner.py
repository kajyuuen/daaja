import random
from typing import List, Tuple

from daaja.ner_sda.labelwise_token_replacement_augmentor import \
    LabelwiseTokenReplacementAugmentor
from daaja.ner_sda.mention_replacement_augmentor import \
    MentionReplacementAugmentor
from daaja.ner_sda.shuffle_within_segments_augmentor import \
    ShuffleWithinSegmentsAugmentor
from daaja.ner_sda.synonym_replacement_augmentor import \
    SynonymReplacementAugmentor
from daaja.ner_sda.utils import get_entity_dict, get_token2prob_in_label
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
                 num_aug: int):
        # setup
        resouces = Resouces()
        entity_dict = get_entity_dict(tokens_list, labels_list)
        token_and_prob_in_label = get_token2prob_in_label(tokens_list, labels_list, p_power=p_power)

        # augmentors
        self.labelwise_token_replacement_augmentor = LabelwiseTokenReplacementAugmentor(token_and_prob_in_label, p=p_lwtr, resouces=resouces)
        self.mention_replacement_augmentor = MentionReplacementAugmentor(entity_dict=entity_dict, p=p_mr)
        self.shuffle_within_segments_augmentor = ShuffleWithinSegmentsAugmentor(p=p_sis)
        self.synonym_replacement_augmentor = SynonymReplacementAugmentor(token_and_prob_in_label=token_and_prob_in_label, p=p_sr, resouces=resouces)

        # params
        self.num_aug = num_aug

    def augments(self, tokens: List[str], labels: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        augmented_tokens_and_labels = []
        num_new_per_technique = int(self.num_aug / 4) + 1

        # Label-wise token replacement (LwTR)
        if self.labelwise_token_replacement_augmentor.p > 0:
            for _ in range(num_new_per_technique):
                aug_tokens, aug_labels = self.labelwise_token_replacement_augmentor.augment(tokens, labels)
                augmented_tokens_and_labels.append([aug_tokens, aug_labels])

        # Synonym replacement (SR)
        if self.labelwise_token_replacement_augmentor.p > 0:
            for _ in range(num_new_per_technique):
                aug_tokens, aug_labels = self.synonym_replacement_augmentor.augment(tokens, labels)
                augmented_tokens_and_labels.append([aug_tokens, aug_labels])

        # Mention replacement (MR)
        if self.labelwise_token_replacement_augmentor.p > 0:
            for _ in range(num_new_per_technique):
                aug_tokens, aug_labels = self.mention_replacement_augmentor.augment(tokens, labels)
                augmented_tokens_and_labels.append([aug_tokens, aug_labels])

        # Shuffle within segments (SiS)
        if self.labelwise_token_replacement_augmentor.p > 0:
            for _ in range(num_new_per_technique):
                aug_tokens, aug_labels = self.shuffle_within_segments_augmentor.augment(tokens, labels)
                augmented_tokens_and_labels.append([aug_tokens, aug_labels])

        random.shuffle(augmented_tokens_and_labels)
        if self.num_aug >= 1:
            augmented_tokens_and_labels = augmented_tokens_and_labels[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_tokens_and_labels)
            augmented_tokens_and_labels = [s for s in augmented_tokens_and_labels if random.uniform(0, 1) < keep_prob]

        # Append the original sentence
        augmented_tokens_and_labels.append([tokens, labels])
        augmented_tokens_list = [tokens_list for tokens_list, _ in augmented_tokens_and_labels]
        augmented_labels_list = [labels_list for _, labels_list in augmented_tokens_and_labels]
        return augmented_tokens_list, augmented_labels_list
