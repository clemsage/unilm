#!/bin/bash

unzip data/SROIE/raw_dataset.zip -d data/SROIE/raw

python SROIE.py --do_pre_processing --raw_dataset_path data/SROIE/raw --dataset_path data/SROIE

python preprocess.py --data_dir data/SROIE/training/annotations --data_split train --output_dir data/SROIE --model_name_or_path bert-base-uncased  --max_len 510

python preprocess.py --data_dir data/SROIE/development/annotations --data_split dev --output_dir data/SROIE --model_name_or_path bert-base-uncased --max_len 510

python preprocess.py --data_dir data/SROIE/test/annotations --data_split test --output_dir data/SROIE --model_name_or_path bert-base-uncased --max_len 510

cat data/SROIE/train.txt | cut -d$'\t' -f 2 | grep -v "^$"| sort | uniq > data/SROIE/labels.txt