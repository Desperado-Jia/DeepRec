#!/usr/bin/env bash

python3 deepfm.py \
--phase "export" \
--model_dir ".//log" \
--export_dir ".//export" \
--task "binary" \
--field_size 13 \
--feat_size 110 \
--embed_size 8 \
--hidden_sizes 32,32,16,8