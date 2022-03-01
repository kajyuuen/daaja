import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from daaja.methods.eda.easy_data_augmentor import EasyDataAugmentor
from tqdm import tqdm


@dataclass
class Data:
    label: str
    sentence: str

    def print_l(self):
        return "{}\t{}".format(self.label, self.sentence)


def main(input_path: Path,
         output_path: Path,
         alpha_sr: float,
         alpha_ri: float,
         alpha_rs: float,
         p_rd: float,
         num_aug: int,
         verbose: bool):
    datum = load_tsv(input_path=input_path)
    eda = EasyDataAugmentor(alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd, num_aug=num_aug, verbose=verbose)

    results = []
    for data in tqdm(datum):
        aug_sentences = eda.augments(data.sentence)
        results.extend([Data(data.label, aug_sentence) for aug_sentence in aug_sentences])

    write_tsv(output_path=output_path, datum=results)


def load_tsv(input_path: Path) -> List[Data]:
    results = []
    with open(input_path, encoding="utf-8", newline='') as f:
        for cols in csv.reader(f, delimiter='\t'):
            results.append(Data(cols[0], cols[1]))
    return results


def write_tsv(output_path: Path, datum: List[Data]):
    with open(output_path, mode="w") as f:
        f.write('\n'.join([data.print_l() for data in datum]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # file
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    # params
    parser.add_argument("--num_aug", default=9, type=int)
    # alpha
    parser.add_argument("--alpha_sr", default=0.1, type=float,
                        help="percent of words in each sentence to be replaced by synonyms")
    parser.add_argument("--alpha_rd", default=0.1, type=float,
                        help="percent of words in each sentence to be deleted")
    parser.add_argument("--alpha_ri", default=0.1, type=float,
                        help="percent of words in each sentence to be inserted")
    parser.add_argument("--alpha_rs", default=0.1, type=float,
                        help="percent of words in each sentence to be swapped")
    # setting
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    main(args.input, args.output, args.alpha_sr, args.alpha_ri, args.alpha_rs, args.alpha_rd, args.num_aug, args.verbose)
