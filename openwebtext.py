# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""The Open WebText Corpus"""

from __future__ import absolute_import, division, print_function
import logging

import os
import re
from itertools import chain
import re
import unicodedata

import datasets

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

_CITATION = """\
@misc{Gokaslan2019OpenWeb,
  title={OpenWebText Corpus},
  author={Aaron Gokaslan*, Vanya Cohen*, Ellie Pavlick, Stefanie Tellex},
  howpublished{\\url{http://Skylion007.github.io/OpenWebTextCorpus}},
  year={2019}
}
"""

_DESCRIPTION = """\
An open-source replication of the WebText dataset from OpenAI.
"""

_URL = "https://zenodo.org/record/3834942/files/openwebtext.tar.xz"

def clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        elif ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            output.append(" ")
        elif is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def is_control(char):
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def is_whitespace(char):
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


class Openwebtext(datasets.GeneratorBasedBuilder):
    """The Open WebText dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=datasets.Version("1.0.0"),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://skylion007.github.io/OpenWebTextCorpus/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        d = "./cache/downloads/extracted/"
        ex_dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
        nested_txt_files = [
            [
                os.path.join(ex_dir, txt_file_name)
                for txt_file_name in sorted(os.listdir(ex_dir))
                if txt_file_name.endswith("txt")
            ]
            for ex_dir in ex_dirs
        ]
        txt_files = chain(*nested_txt_files)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"txt_files": txt_files}),
        ]

    def _generate_examples(self, txt_files):
        """ Yields examples. """
        for idx, filepath in enumerate(txt_files):
            with open(filepath, encoding="utf-8") as f:
                sentence = re.sub("\n+", "\n", f.read()).strip()
                sentence = re.sub("(https?|http|ftp|file):\/\/[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", sentence)
                sentence = clean_text(sentence)
                sentence = re.sub(" +", " ", sentence).strip()
                yield idx, {"text":sentence}
