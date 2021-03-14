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

import os
import re
from itertools import chain

import datasets


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
                yield idx, {"text": re.sub("\n+", "\n", f.read()).strip()}

