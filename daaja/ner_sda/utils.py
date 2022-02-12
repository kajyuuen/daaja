from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

EntityDict = Dict[str, List[str]]
TokenAndProbInLabel = DefaultDict[str, Tuple[List[str], List[float]]]


def get_entity_dict(tokens_list: List[List[str]],
                    labels_list: List[List[str]]) -> EntityDict:
    splitter = " "
    entities: DefaultDict[str, List[str]] = defaultdict(list)
    for tokens, labels in zip(tokens_list, labels_list):
        entity = []
        current_label = None
        for token, label in zip(tokens, labels):
            if label == "O":
                entities["O"].append(token)
                if current_label is not None:
                    entities[current_label].append(f"{splitter}".join(entity))
                entity = []
                current_label = None
            if label[0].startswith("B"):
                current_label = label[2:]
                entity.append(token)
            elif label[0].startswith("I"):
                entity.append(token)
        if current_label is not None:
            entities[current_label].append(f"{splitter}".join(entity))
    entity_dict = {k: list(set(v)) for k, v in entities.items() for vi in v}
    return entity_dict


def get_token2prob_in_label(tokens_list: List[List[str]],
                            labels_list: List[List[str]],
                            p_power: float = 1) -> TokenAndProbInLabel:
    counter: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tokens, labels in zip(tokens_list, labels_list):
        for token, label in zip(tokens, labels):
            counter[label][token] += 1

    token_and_prob_in_label: DefaultDict[str, Tuple[List[str], List[float]]] = defaultdict(tuple)
    for label, counter_i in counter.items():
        sum_n = sum(counter_i.values())
        tokens, probs = [], []
        for token, cnt in counter_i.items():
            prob = np.power(cnt, p_power) / sum_n
            tokens.append(token)
            probs.append(prob)
        token_and_prob_in_label[label] = (tokens, probs)
    return token_and_prob_in_label
