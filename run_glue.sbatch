#!/bin/bash

#SBATCH -J albert-base-v2-liyihong                         # 作业名
#SBATCH -o joblog/R-%x.%j.out               # 屏幕上的输出文件重定向
#SBATCH -e joblog/R-%x.%j.err               # 错误

#SBATCH -p gpu_v100               # 作业提交的分区
#SBATCH -N 1                      # 作业申请节点数
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数

# 设置运行环境
module load anaconda3
module load cudnn/7.6.4-CUDA10.1    #implicitly load a specific CUDA version
module load proxy
source activate thesis-lyh
export HTTP_PROXY=http://10.20.18.21:3128
export HTTPS_PROXY=http://10.20.18.21:3128

# 输入要执行的命令
export TASK_NAME=MRPC
export MODEL_NAME=albert-base-v2

python run_glue.py \
  --model_name_or_path $MODEL_NAME \
  --train_file datasets/$TASK_NAME/train.tsv \
  --validation_file datasets/$TASK_NAME/dev.tsv \
  --test_file datasets/$TASK_NAME/test.tsv \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --n_gpu 4 \
  --output_dir output/glue/$TASK_NAME/$MODEL_NAME/