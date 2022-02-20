#!/bin/bash

for base_num in 500; do
    subset_path="./subsets/${base_num}_train.tsv"
    for p in 0.1 0.3 0.5 0.7; do
        for num_aug in 1 3 6 10; do
            command="python -m daaja.ner_sda.run --input ${subset_path} --output ./subsets/aug_${base_num}_p_${p}_num_${num_aug}.tsv \
                                                    --p_lwtr ${p} --p_mr ${p} --p_sis ${p} --p_sr ${p} --num_aug ${num_aug}"
            echo $command
            eval $command
        done
    done
done
