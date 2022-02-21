import random
from typing import List, Tuple

import numpy as np
from daaja.augmentors.sequence_labeling.sequence_labeling_augmentor import \
    SequenceLabelingAugmentor


class ShuffleWithinSegmentsAugmentor(SequenceLabelingAugmentor):
    def __init__(self,
                 p: float = 0.1) -> None:
        self.p = p

    def augment(self,
                tokens: List[str],
                labels: List[str]) -> Tuple[List[str], List[str]]:
        generated_tokens, generated_labels = [], []
        shuffled_idx = self._shuffle_within_segments(labels)
        assert len(shuffled_idx) == len(labels)
        for i in shuffled_idx:
            generated_token = tokens[i]
            generated_label = labels[i]
            generated_tokens.append(generated_token)
            generated_labels.append(generated_label)
        return generated_tokens, generated_labels

    def _shuffle_within_segments(self, labels: List[str]) -> List[int]:
        segments = [0]
        for i, label in enumerate(labels):
            if i == 0:
                continue
            if label == "O":
                if labels[i - 1] == "O":
                    segments.append(segments[-1])
                else:
                    segments.append(segments[-1] + 1)
            elif label.startswith("B"):
                segments.append(segments[-1] + 1)
            else:
                segments.append(segments[-1])

        shuffled_idx = []
        start, end = 0, 0
        while start < len(segments) and end < len(segments):
            while end < len(segments) and segments[end] == segments[start]:
                end += 1
            segment = [i for i in range(start, end)]
            if len(segment) > 1 and np.random.binomial(1, self.p, 1)[0] == 1:
                random.shuffle(segment)
            shuffled_idx += segment
            start = end
        return shuffled_idx
