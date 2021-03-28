#!/bin/bash

python train.py \
    --force \
    --student_type albert \
    --student_config training_configs/albert-base-v2-config.json \
    --teacher_type albert \
    --teacher_name albert-xlarge-v2 \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path output/xxlarge-2048 \
    --data_file datasets/binarized_text.albert.pickle \
    --token_counts datasets/token_counts.albert.pickle
    # --force # overwrites the `dump_path` if it already exists.
