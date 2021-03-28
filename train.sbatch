#!/bin/bash

export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_PORT=10086
export MASTER_ADDR="127.0.0.1"

pkill -f 'python -u train.py'

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
        --force \
        --n_gpu $WORLD_SIZE \
        --student_type albert \
        --student_config training_configs/albert-base-v2-config.json \
        --teacher_type albert \
        --teacher_name albert-xlarge-v2 \
        --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --alpha_clm 0.0 --mlm \
        --freeze_pos_embs \
        --dump_path output/xxlarge-2048 \
        --data_file datasets/binarized_text.albert.v2.pickle \
        --token_counts datasets/token_counts.albert.v2.pickle
