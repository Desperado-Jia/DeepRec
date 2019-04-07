#!/usr/bin/env bash

python3 deepfm.py \
--phase "eval" \
--model_dir ".//log" \
--data_dir "..//..//examples//Adventure-Works-Cycles" \
--delimiter " " \
--separator ":" \
--batch_size 64 \
--task "binary" \
--field_size 13 \
--feat_size 110 \
--embed_size 8 \
--hidden_sizes 32,32,16,8 \
--lamb 0.001
