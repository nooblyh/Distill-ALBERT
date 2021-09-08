#!/bin/bash

#SBATCH -J lyh-glue                         # 作业名
#SBATCH -o joblog/R-%x.%j.out               # 屏幕上的输出文件重定向

#SBATCH -p gpu_v100               # 作业提交的分区
#SBATCH -N 1                      # 作业申请节点数
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数

# distilbert-base-uncased, bert-base-uncased, albert-base-v2
# 设置运行环境
module load anaconda3

module load cudnn/7.6.4-CUDA10.1    #implicitly load a specific CUDA version
module load proxy
source activate thesis-lyh

# 输入要执行的命令
export TASK_NAME=sst2
export MODEL_NAME=correct
export MODEL_PATH=output/correct

CUDA_VISIBLE_DEVICES=0,1,2,3 ~/.conda/envs/thesis-lyh/bin/python run_glue.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --cache_dir models \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir output/glue/$TASK_NAME/$MODEL_NAME-4.5/
