import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from daaja.methods.ner_sda.simple_data_augmentation_for_ner import \
    SimpleDataAugmentationforNER
from tqdm import tqdm


@dataclass
class Data:
    label: str
    sentence: str

    def print_l(self):
        return "{}\t{}".format(self.label, self.sentence)


def main(input_path: Path,
         output_path: Path,
         p_power: float,
         p_lwtr: float,
         p_mr: float,
         p_sis: float,
         p_sr: float,
         num_aug: int,
         verbose: bool):
    tokens_list, labels_list = load_conll(input_path=input_path)
    sda = SimpleDataAugmentationforNER(tokens_list=tokens_list, labels_list=labels_list, p_power=p_power,
                                       p_lwtr=p_lwtr, p_mr=p_mr, p_sis=p_sis, p_sr=p_sr, num_aug=num_aug, verbose=verbose)

    result_tokens_list, result_labels_list = [], []
    for tokens, labels in tqdm(zip(tokens_list, labels_list)):
        arg_tokens_list, arg_labels_list = sda.augments(tokens, labels)
        result_tokens_list.extend(arg_tokens_list)
        result_labels_list.extend(arg_labels_list)

    write_conll(output_path=output_path, tokens_list=result_tokens_list, labels_list=result_labels_list)


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
                tokens_list: List[List[str]],
                labels_list: List[List[str]]):
    text = ""
    for tokens, labels in zip(tokens_list, labels_list):
        for token, label in zip(tokens, labels):
            text += "{}\t{}\n".format(token, label)
        text += "\n"

    with open(output_path, mode="w") as f:
        f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # file
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    # params
    parser.add_argument("--num_aug", default=4, type=int)
    # alpha
    parser.add_argument("--p_power", default=1, type=float,
                        help="p_power hyper params")
    parser.add_argument("--p_lwtr", default=0.1, type=float,
                        help="percent of words in each sentence to be labelwise token replaced")
    parser.add_argument("--p_mr", default=0.1, type=float,
                        help="percent of words in each sentence to be mention replaced")
    parser.add_argument("--p_sis", default=0.1, type=float,
                        help="percent of words in each sentence to be shuffle")
    parser.add_argument("--p_sr", default=0.1, type=float,
                        help="percent of words in each sentence to be replaced by synonyms")
    # setting
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    main(args.input, args.output, args.p_power, args.p_lwtr, args.p_mr, args.p_sis, args.p_sr, args.num_aug, args.verbose)
