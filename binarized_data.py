# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before distillation.
"""
import argparse
import logging
import pickle
import random
import time
import re
import unicodedata
from itertools import chain

import numpy as np

from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer, AlbertTokenizer
from datasets import load_dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--tokenizer_type", type=str, default="bert", choices=["bert", "roberta", "gpt2", "albert"])
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="The tokenizer to use.")
    parser.add_argument("--dump_file", type=str, default="data/dump", help="The dump file prefix.")
    args = parser.parse_args()

    '''
    bookcorpus = load_dataset('./bookcorpus.py', cache_dir='cache')
    train_bookcorpus = bookcorpus["train"]
    '''
    wikipedia = load_dataset('./wikipedia.py', '20200501.en', beam_runner='DirectRunner', cache_dir='cache')
    train_wikipedia = wikipedia["train"]


    logger.info(f"Loading Tokenizer ({args.tokenizer_name})")
    if args.tokenizer_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map["cls_token"]  # `[CLS]`
        sep = tokenizer.special_tokens_map["sep_token"]  # `[SEP]`
    elif args.tokenizer_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map["cls_token"]  # `<s>`
        sep = tokenizer.special_tokens_map["sep_token"]  # `</s>`
    elif args.tokenizer_type == "albert":
        tokenizer = AlbertTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map["cls_token"]
        sep = tokenizer.special_tokens_map["sep_token"]
    elif args.tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
        sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`

    logger.info("Start encoding")
    logger.info(f"{len(train_wikipedia)} wikipedia examples to process.")

    rslt = []
    iter = 0
    interval = 10000
    start = time.time()
    for s in train_wikipedia:
        sentence = s["text"]
        sentence = f"{bos} {sentence.strip()} {sep}"
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        if len(token_ids) <= 11:
            continue
        rslt.append(np.uint16(token_ids))
        iter += 1
        if iter % interval == 0:
            end = time.time()
            logger.info(f"{iter} examples processed. - {(end-start):.2f}s/{interval}expl")
            start = time.time()
    logger.info("Finished binarization")
    logger.info(f"{iter} examples processed.")

    dp_file = f"{args.dump_file}.{args.tokenizer_type}.pickle"
    logger.info(f"Dump to {dp_file}")
    with open(dp_file, "wb") as handle:
        pickle.dump(rslt, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
