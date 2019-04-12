#!/usr/bin/env bash

python3 deepfm.py \
--phase "train" \
--perform_valid_during_train True \
--model_dir ".//log" \
--data_dir "..//..//examples//Adventure-Works-Cycles" \
--delimiter " " \
--separator ":" \
--batch_size 64 \
--epochs 10 \
--shuffle=True \
--task "binary" \
--field_size 13 \
--feat_size 110 \
--embed_size 8 \
--hidden_sizes 32,32,16,8 \
--dropouts 0.3,0.3,0.2,0.1 \
--lamb 0.001 \
--optimizer "adam" \
--learning_rate 0.0001