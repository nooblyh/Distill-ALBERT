#!/bin/bash
#SBATCH -J lyh                              # 作业名
#SBATCH -o token_counts.out                  # 屏幕上的输出文件重定向

#SBATCH -p gpu_v100                   # 作业提交的分区
#SBATCH -N 1                      # 作业申请节点数
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数

source env.sh
ion

python token_counts.py \
    --data_file datasets/wikipedia-binarized_text.albert.pickle \
    --token_counts_dump datasets/wikipedia-token_counts.albert.pickle \
    --vocab_size 30000

