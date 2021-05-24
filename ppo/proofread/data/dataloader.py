from typing import List
import os
import numpy as np
from transformers import BertTokenizer
import requests
import time
from nltk.tokenize import sent_tokenize

from ..noisy import make_noisy
from ..utils import shuffle


def get_web_text(url) -> str:
    while True:
        try:
            r = requests.get(url)
        except requests.exceptions.ConnectionError as e:
            # if server is dead
            print(url, '\t', e)
            time.sleep(600)
    return r.content.decode('utf8')


class DataLoader:
    def __init__(self, tokenizer: BertTokenizer, base_url='http://2runo.com:13000/wikicorpus/'):
        self.base_url = base_url
        self.n_range = list(map(int, get_web_text(self.base_url+'status.txt').split('-')))
        self.n_range[1] += 1

        self.data: List[str] = []

        self.tokenizer = tokenizer

    def load_file(self) -> None:
        # Load randomly selected text file.
        del self.data

        # load two files
        url = self.base_url + f'{n}.txt'
        data: str = get_web_text(url)
        self.data = shuffle(data.split('\n'))

    def get_n(self, n: int) -> List[str]:
        # Return n sentences from `self.data`
        if len(self.data) < n:
            # if data is insufficient, load data again.
            r = self.data.copy()
            self.load_file()
            remain = n - len(r)
            r.extend(self.data[:remain])
            self.data = self.data[remain:]
        else:
            r = self.data[:n]
            self.data = self.data[n:]
        return r

    @staticmethod
    def split_sentence(sent: str) -> List[str]:
        # Tokenize a sentence into a list of words.
        return sent.split(' ')

    @staticmethod
    def preprocessing(sent: str) -> str:
        # Preprocess a sentence
        sent = sent.lower()  # make uncased
        return sent

    def get_batch(self, batch_size: int) -> (List[List[int]], List[List[int]], np.ndarray):
        # Return preprocessed batch data.
        # inputs: [A,B,..,C,D, E,F,...,G,H]
        # labels: [0,0,...,0,0,1,1,...,1,1]
        # A~D : tools texts -> label 0
        # E~H : original texts -> label 1
        assert batch_size % 2 == 0  # batch size must be even number.

        half = batch_size // 2
        x1: List[str] = self.get_n(half)  # get data
        x1 = [self.preprocessing(sent) for sent in x1]
        x0: List[List[str]] = [make_noisy(self.split_sentence(sent)) for sent in x1]  # tools
        x0 = [i for i in x0 if i is not None]
        x0: List[str] = [' '.join(words) for words in x0]
        # x0 : label 0, x1 : label 1

        labels: np.ndarray = np.concatenate([np.zeros((len(x0), 1)), np.ones((len(x1), 1))])
        encoding = self.tokenizer(x0 + x1, padding=True, truncation=True, max_length=100)
        input_ids: List[List[int]] = encoding['input_ids']
        attention_mask: List[List[int]] = encoding['attention_mask']

        del x1, x0

        return input_ids, attention_mask, labels
