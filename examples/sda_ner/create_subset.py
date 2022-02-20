import os
import random
from pathlib import Path
from typing import List, Tuple


def load_conll(input_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    tokens_list, labels_list = [], []
    tokens, labels = [], []
    for line in input_path.read_text().split("\n"):
        cols = line.split("\t")
        if len(cols) != 2:
            if len(tokens) > 0:
                tokens_list.append(tokens)
                labels_list.append(labels)
                tokens, labels = [], []
            continue
        tokens.append(cols[0])
        labels.append(cols[1])
    if len(tokens) > 0:
        tokens_list.append(tokens)
        labels_list.append(labels)
    return tokens_list, labels_list


def write_conll(output_path: Path,
                datasets: List[List[List[str]]]):
    text = ""
    for dataset in datasets:
        tokens, labels = dataset
        for token, label in zip(tokens, labels):
            text += "{}\t{}\n".format(token, label)
        text += "\n"

    with open(output_path, mode="w") as f:
        f.write(text)


def create_subset(ns: List[int], output_dir: Path):
    tokens_list, labels_list = load_conll(Path("./dataset/ja.wikipedia.conll"))

    os.makedirs(output_dir, exist_ok=True)

    datasets = [[tokens, labels] for tokens, labels in zip(tokens_list, labels_list)]

    random.shuffle(datasets)
    all_n = len(datasets)
    max_n = max(ns)
    n_val = int(0.5 * (all_n - max_n))
    dataset_val = datasets[max_n:max_n + n_val]
    dataset_test = datasets[max_n + n_val:]
    write_conll(output_dir / Path("valid.tsv"), dataset_val)
    write_conll(output_dir / Path("test.tsv"), dataset_test)

    for n_train in ns:
        dataset_train = datasets[:n_train]
        write_conll(output_dir / Path(f"{n_train}_train.tsv"), dataset_train)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", nargs='+', required=True, type=int)
    parser.add_argument("-o", required=True, type=Path)
    args = parser.parse_args()
    create_subset(args.ns, args.o)
