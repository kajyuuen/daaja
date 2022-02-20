# SDA example

## Download dataset

```
wget -P dataset/ https://github.com/Hironsan/IOB2Corpus/raw/master/ja.wikipedia.conll
```

## Create subset

```
python create_subset.py -n 50 150 500 -o subsets
./data_augment.sh
```

## Run model

Look at `./SimpleDataAugmentationforNamedEntityRecognition_example.ipynb`
