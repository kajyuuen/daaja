# EDA example

## Download dataset

```
wget -P dataset/ https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar -zxf dataset/ldcc-20140209.tar.gz -C dataset/
```

## Create subset

```
python create_subset.py -n 500 2000 5000 -o subsets
python -m daaja.eda.run --input subsets/500_train.tsv --output subsets/aug_500_train.tsv --alpha_sr 0.05 --alpha_rd 0.05 --alpha_ri 0.05 --alpha_rs 0.05 --num_aug 16
python -m daaja.eda.run --input subsets/2000_train.tsv --output subsets/aug_2000_train.tsv --alpha_sr 0.05 --alpha_rd 0.05 --alpha_ri 0.05 --alpha_rs 0.05 --num_aug 8
python -m daaja.eda.run --input subsets/5000_train.tsv --output subsets/aug_5000_train.tsv --alpha_sr 0.1 --alpha_rd 0.1 --alpha_ri 0.1 --alpha_rs 0.1 --num_aug 4
```

## Run model

Look at `./EasyDataAugmentation_example.ipynb`
