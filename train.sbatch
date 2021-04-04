#!/bin/bash

#SBATCH -J newlyh-node1                        # 作业名
#SBATCH -o joblog/R-%x.%j.out               # 屏幕上的输出文件重定向

#SBATCH -p gpu_v100               # 作业提交的分区
#SBATCH -N 1                      # 作业申请节点数
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数

# 设置运行环境
module load anaconda3
module load cudnn/7.6.4-CUDA10.1    #implicitly load a specific CUDA version
module load proxy
source activate thesis-lyh

function ion {
 
        export http_proxy=http://10.20.18.21:3128
        export HTTP_PROXY=http://10.20.18.21:3128
        export https_proxy=http://10.20.18.21:3128
        export HTTPS_PROXY=http://10.20.18.21:3128
 
        export no_proxy="localhost,127.0.0.1"
 
        git config --global http.proxy http://10.20.18.21:3128
        git config --global https.proxy http://10.20.18.21:3128
 
}

ion
export NODE_RANK=1
export N_NODES=2

export N_GPU_NODE=4
export WORLD_SIZE=8
export MASTER_PORT=10086
export MASTER_ADDR="localhost"

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
        --dump_path output/final \
        --data_file datasets/binarized_text.albert.v2.pickle \
        --token_counts datasets/token_counts.albert.v2.pickle
