#!/bin/bash

python binarized_data.py \
    --tokenizer_type albert \
    --tokenizer_name ./albert-base-v2/ \
    --dump_file datasets/binarized_text
