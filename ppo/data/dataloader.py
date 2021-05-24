from typing import List
import os
import numpy as np
import requests
import time


def shuffle(x: list) -> list:
    np.random.shuffle(x)
    return x


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
    def __init__(self, base_url='http://2runo.com:13000/wikicorpus/'):
        self.base_url = base_url
        self.n_range = list(map(int, get_web_text(self.base_url+'status.txt').split('-')))
        self.n_range[1] += 1
        self.data: List[str] = []

    def load_file(self) -> None:
        # Load randomly selected text file.
        del self.data

        n = np.random.choice(range(*self.n_range))
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

    def split_sentence(self, sent: str) -> List[str]:
        # Tokenize a sentence into a list of words.
        return sent.split(' ')

    @staticmethod
    def preprocessing(sent: str) -> str:
        # Preprocess a sentence
        sent = sent.lower()  # make uncased
        return sent
