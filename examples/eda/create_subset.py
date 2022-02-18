import glob
import os
import random
from pathlib import Path
from typing import List


def create_subset(ns: List[int], output_dir: Path):
    category_list = [
        'dokujo-tsushin',
        'it-life-hack',
        'kaden-channel',
        'livedoor-homme',
        'movie-enter',
        'peachy',
        'smax',
        'sports-watch',
        'topic-news'
    ]

    datasets = []
    for i, category in enumerate(category_list):
        for file in glob.glob(f'./dataset/text/{category}/{category}*.txt'):
            lines = open(file).read().splitlines()
            text = '\n'.join(lines[3:]).replace("\n", " ")
            datasets.append("{}\t{}".format(i, text))

    os.makedirs(output_dir, exist_ok=True)

    random.shuffle(datasets)
    all_n = len(datasets)
    max_n = max(ns)
    n_val = int(0.5 * (all_n - max_n))
    dataset_val = datasets[max_n:max_n + n_val]
    dataset_test = datasets[max_n + n_val:]
    with open(output_dir / Path("valid.tsv"), mode='w') as f:
        print(f"Valid size: {len(dataset_val)}")
        f.write("\n".join(dataset_val))
    with open(output_dir / Path("test.tsv"), mode='w') as f:
        print(f"Test size: {len(dataset_test)}")
        f.write("\n".join(dataset_test))

    for n_train in ns:
        dataset_train = datasets[:n_train]
        with open(output_dir / Path(f"{n_train}_train.tsv"), mode='w') as f:
            f.write("\n".join(dataset_train))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", nargs='+', required=True, type=int)
    parser.add_argument("-o", required=True, type=Path)
    args = parser.parse_args()
    create_subset(args.ns, args.o)
