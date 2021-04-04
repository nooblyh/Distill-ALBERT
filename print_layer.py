from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import os

config = AutoConfig.from_pretrained (
    "output/xlarge-2048-2nodes/",
    num_labels=1,
    finetuning_task='qqp'
)
tokenizer = AutoTokenizer.from_pretrained (
    "output/xlarge-2048-2nodes/",
    use_fast=True
)

state_dict = torch.load(os.path.join("output/xlarge-2048-2nodes/", "checkpoint.pth"), map_location=torch.device('cpu'))
model = AutoModelForSequenceClassification.from_pretrained (
    pretrained_model_name_or_path=None,
    config=config,
    state_dict=state_dict
)

for name, _ in model.named_parameters():
   print(name)
