from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification
)

import torch
import os

config = AutoConfig.from_pretrained (
    "training_configs/albert-base-v2-config.json",
    num_labels=1,
    finetuning_task='qqp'
)

model = AutoModelForSequenceClassification.from_pretrained (
    pretrained_model_name_or_path="albert-base-v2",
    config=config
)

for name, _ in model.named_parameters():
   print(name)
