# daaja

This repository has a implementations of data augmentation for NLP for Japanese:

- [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)
- [An Analysis of Simple Data Augmentation for Named Entity Recognition](https://arxiv.org/abs/2010.11683)

## Install

```
pip install daaja
```

## How to use

### [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)

#### Command

```sh
python -m daaja.eda.run --input input.tsv --output data_augmentor.tsv
```

The format of input.tsv is as follows:

```tsv
1	この映画はとてもおもしろい
0	つまらない映画だった
```

#### In Python

```python
from daaja.eda import EasyDataAugmentor
augmentor = EasyDataAugmentor(alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4)
text = "日本語でデータ拡張を行う"
aug_texts = augmentor.augments(text)
print(aug_texts)
# ['日本語でを拡張データ行う', '日本語でデータ押広げるを行う', '日本語でデータ拡張を行う', '日本語で智見拡張を行う', '日本語でデータ拡張を行う']
```

### [An Analysis of Simple Data Augmentation for Named Entity Recognition](https://arxiv.org/abs/2010.11683)

#### Command

```sh
python -m daaja.ner_sda.run --input input.tsv --output data_augmentor.tsv
```

The format of input.tsv is as follows:

```tsv
私	O
は	O
田中	B-PER
と	O
いい	O
ます	O
```

#### In Python

```python
from daaja.ner_sda import SimpleDataAugmentationforNER
tokens_list = [
    ["私", "は", "田中", "と", "いい", "ます"],
    ["筑波", "大学", "に", "所属", "して", "ます"],
    ["今日", "から", "筑波", "大学", "に", "通う"],
    ["茨城", "大学"],
]
labels_list = [
    ["O", "O", "B-PER", "O", "O", "O"],
    ["B-ORG", "I-ORG", "O", "O", "O", "O"],
    ["B-DATE", "O", "B-ORG", "I-ORG", "O", "O"],
    ["B-ORG", "I-ORG"],
]
augmentor = SimpleDataAugmentationforNER(tokens_list=tokens_list, labels_list=labels_list,
                                            p_power=1, p_lwtr=1, p_mr=1, p_sis=1, p_sr=1, num_aug=4)
tokens = ["吉田", "さん", "は", "株式", "会社", "A", "に", "出張", "予定", "だ"]
labels = ["B-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O"]
augmented_tokens_list, augmented_labels_list = augmentor.augments(tokens, labels)
print(augmented_tokens_list)
# [['吉田', 'さん', 'は', '株式', '会社', 'A', 'に', '出張', '志す', 'だ'],
#  ['吉田', 'さん', 'は', '株式', '大学', '大学', 'に', '出張', '予定', 'だ'],
#  ['吉田', 'さん', 'は', '株式', '会社', 'A', 'に', '出張', '予定', 'だ'],
#  ['吉田', 'さん', 'は', '筑波', '大学', 'に', '出張', '予定', 'だ'],
#  ['吉田', 'さん', 'は', '株式', '会社', 'A', 'に', '出張', '予定', 'だ']]
print(augmented_labels_list)
# [['B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O'],
#  ['B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O'],
#  ['B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O'],
#  ['B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O'],
#  ['B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O']]
```

**Reference**

- https://arxiv.org/abs/1901.11196
- https://arxiv.org/abs/1901.11196
- https://github.com/jasonwei20/eda_nlp
- https://qiita.com/tchih11/items/aef9505d26d1bf06a04c
- https://user-first.ikyu.co.jp/entry/2021/07/27/155513
