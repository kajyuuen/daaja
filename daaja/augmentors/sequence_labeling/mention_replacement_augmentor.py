from typing import List, Tuple

import numpy as np
from daaja.augmentors.sequence_labeling.sequence_labeling_augmentor import \
    SequenceLabelingAugmentor
from daaja.augmentors.sequence_labeling.utils import EntityDict


class MentionReplacementAugmentor(SequenceLabelingAugmentor):
    def __init__(self,
                 entity_dict: EntityDict,
                 p: float = 0.1) -> None:
        self.p = p
        self.entity_dict = entity_dict

    def augment(self,
                tokens: List[str],
                labels: List[str]) -> Tuple[List[str], List[str]]:
        generated_tokens, generated_labels = [], []
        for i, (label, token) in enumerate(zip(labels, tokens)):
            if label == "O":
                generated_tokens_i = [token]
                generated_labels_i = [label]
            elif label.startswith("B"):
                label_type = label[2:]
                is_change_mention = np.random.binomial(1, self.p, 1)[0]
                if is_change_mention:
                    candidates = self.entity_dict[label_type]
                    random_idx = np.random.choice(len(candidates), 1)[0]
                    replaced_mention = candidates[random_idx].split()
                    generated_tokens_i = replaced_mention
                    generated_labels_i = ["B-" + label_type if i == 0 else "I-" + label_type for i in range(len(replaced_mention))]
                else:
                    generated_tokens_i = [token]
                    generated_labels_i = [label]
                    next_idx = i + 1
                    while next_idx < len(tokens) and tokens[next_idx].startswith("I"):
                        generated_tokens_i.append(tokens[next_idx])
                        generated_labels_i.append(labels[next_idx])
                        next_idx += 1
            elif label[0] == "I":
                continue
            else:
                raise ValueError("unreachable line...")
            generated_tokens.extend(generated_tokens_i)
            generated_labels.extend(generated_labels_i)
        return generated_tokens, generated_labels
