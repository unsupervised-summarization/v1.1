from typing import List
import numpy as np
from transformers import GPT2Tokenizer
import requests
import time
from nltk.tokenize import sent_tokenize

from .utils.sent import extract_sent
from .utils import shuffle


def get_web_text(url: object) -> object:
    first = True
    while True:
        try:
            r = requests.get(url)
            break
        except requests.exceptions.ConnectionError as e:
            # if server is dead
            print(url, '\t', e)
            if first:
                time.sleep(1)
            else:
                time.sleep(60)
                first = False
    return r.content.decode('utf8')


class DataLoader:
    def __init__(self, tokenizer: GPT2Tokenizer, base_url='http://2runo.com:13000/wikicorpus/'):
        self.base_url = base_url
        self.n_range = list(map(int, get_web_text(self.base_url + 'status.txt').split('-')))
        self.n_range[1] += 1
        self.n_range[1] -= 2  # the last two files are used for test.

        self.data: List[str] = []

        self.tokenizer = tokenizer

    def load_file(self) -> None:
        # Load randomly selected text file.
        del self.data

        # load a file
        n = np.random.randint(*self.n_range)
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
    def split_into_sentence(text: str) -> List[str]:
        # split text into a sequence of sentences.
        # ex) f("Hello World. It's good to see you. Thanks for buying this book.")
        # -> ['Hello World.', "It's good to see you.", 'Thanks for buying this book.']
        return sent_tokenize(text)

    @staticmethod
    def preprocessing(sent: str) -> str:
        # Preprocess a sentence
        sent = sent.lower()  # make uncased
        return sent

    def get_batch(self, batch_size: int) -> (List[List[int]], List[List[int]]):
        # Return preprocessed batch data.

        summaries_tokens = []
        documents_tokens = []
        for text in [self.preprocessing(sent) for sent in self.get_n(batch_size)]:
            sents: List[str] = self.split_into_sentence(text)  # the sequence of sentences
            document, summary = extract_sent(sents)  # -> (str, str)
            if document is None:
                continue
            summaries_tokens.append(self.tokenizer.encode(summary))
            documents_tokens.append(self.tokenizer.encode(document))

        return summaries_tokens, documents_tokens
