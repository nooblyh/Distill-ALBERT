#!/bin/bash

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
module load anaconda3
conda info --envs

module load cudnn/7.6.4-CUDA10.1    #implicitly load a specific CUDA version
module load proxy
source activate thesis-lyh
conda list
export HTTP_PROXY=http://10.20.18.21:3128
export HTTPS_PROXY=http://10.20.18.21:3128

# 输入要执行的命令
export TASK_NAME=QQP
export MODEL_NAME=distilbert-base-uncased
which python

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_glue.py \
  --model_name_or_path $MODEL_NAME \
  --train_file datasets/$TASK_NAME/train.csv \
  --validation_file datasets/$TASK_NAME/dev.csv \
  --do_train \
  --do_eval \
  --cache_dir models \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir output/glue/$TASK_NAME/$MODEL_NAME/
