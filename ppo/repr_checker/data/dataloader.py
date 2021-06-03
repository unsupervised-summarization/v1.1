from typing import List
import numpy as np
from transformers import GPT2Tokenizer
import requests
import time
from nltk.tokenize import sent_tokenize

from ..tools import extract_sent
from ..utils import shuffle


def get_web_text(url) -> str:
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
        if self.tokenizer.cls_token_id is None:
            # add [CLS] token
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            # To call `resize_token_embeddings(len(tokenizer))` is required for user.

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

    def get_batch(self, batch_size: int) -> (List[List[int]], List[List[int]], np.ndarray):
        # Return preprocessed batch data.
        # inputs: [A,B,..,C,D, E,F,...,G,H]
        # labels: [0,0,...,0,0,1,1,...,1,1]
        # A~D : real inner sentence -> label 0
        # E~H : fake inner sentence -> label 1
        assert batch_size % 2 == 0  # batch size must be even number.

        half = batch_size // 2
        x1: List[str] = []  # real inner sentence
        for text in [self.preprocessing(sent) for sent in self.get_n(half)]:
            sents: List[str] = self.split_into_sentence(text)  # the sequence of sentences
            text, extracted_sentence = extract_sent(sents)  # -> (str, str)
            if text is None:
                continue
            input_text: str = extracted_sentence + ' [CLS] ' + text
            x1.append(input_text)

        x0: List[str] = []  # fake inner sentence
        temp: List[str] = [self.preprocessing(sent) for sent in self.get_n(half)]
        for text in [self.preprocessing(sent) for sent in self.get_n(half)]:
            sents: List[str] = self.split_into_sentence(text)  # the sequence of sentences
            text, _ = extract_sent(sents)  # -> (str, str)
            if text is None:
                continue

            idx = np.random.randint(0, len(temp))
            temp_sents: List[str] = self.split_into_sentence(temp[idx])  # the sequence of sentences
            del temp[idx]
            _, extracted_sentence = extract_sent(temp_sents, ignore_short=True)  # -> (str, str)

            input_text: str = extracted_sentence + ' [CLS] ' + text
            x0.append(input_text)

        # x0 : label 0, x1 : label 1
        labels: np.ndarray = np.concatenate([np.zeros((len(x0), 1)), np.ones((len(x1), 1))])
        encoding = self.tokenizer(x0 + x1, padding=True, truncation=True, max_length=1000, return_tensors='pt')
        input_ids: List[List[int]] = encoding['input_ids']
        attention_mask: List[List[int]] = encoding['attention_mask']

        del x0, x1

        return input_ids, attention_mask, labels
