import pathlib as pt
from typing import List, Dict

import numpy as np


def tokenize(path: pt.Path, word_sep=" ", statment_sep = "\n",):
    tokens = []
    with open(path, "r") as file:
        for line in file.readlines():
            line = line
            sentence = []
            for word in line.split(" "):
                sentence.append([letter for letter in word])
            tokens.append(sentence)
    return tokens

def vec_trainer(words: List[str]):
    index = {}
    for i, word in enumerate(words):
        index[word] = i
    return index

def word2vec(index: Dict[str, int], words: List[str]):
    result = []
    for word in words:
        result.append(index[])