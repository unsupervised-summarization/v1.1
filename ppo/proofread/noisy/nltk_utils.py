from typing import List
import numpy as np
import os

from nltk.corpus import wordnet
from pyinflect import getInflection


class RandomWord:
    def __init__(self, assets_path='proofread/assets'):
        if not os.path.isdir(assets_path):
            assets_path = 'ppo/' + assets_path
        self.words = self.read_text(assets_path + '/words.txt')
        self.prepositions = self.read_text(assets_path + '/prepositions.txt')
        self.articles = self.read_text(assets_path + '/articles.txt')
        self.stopwords = self.read_text(assets_path + '/stopwords.txt')

    @staticmethod
    def read_text(path: str) -> List[str]:
        with open(path, 'r', encoding='utf8') as f:
            raw = f.read()
        return raw.split('\n')

    def word(self) -> str:
        # return a word randomly
        return np.random.choice(self.words)

    def preposition(self) -> str:
        # return a preposition randomly
        return np.random.choice(self.prepositions)

    def article(self) -> str:
        # return an article randomly
        return np.random.choice(self.articles)

    def stopword(self) -> str:
        # return an stop word randomly
        return np.random.choice(self.stopwords)


def get_syn(word: str) -> str:
    # Return a Synonym
    candidates = [i.name().split('.')[0].replace('_', ' ') for i in wordnet.synsets(word)]
    candidates = [cand for cand in candidates if cand != word]
    if len(candidates) > 0:
        return np.random.choice(candidates)


def to_ing(word: str) -> str:
    # Change a word to a gerund
    word = lemma(word)  # lemmatization
    out = getInflection(word, 'VBG')
    if out is not None:
        return out[0]


def lemma(word: str) -> str:
    # Lemmatize a word
    try:
        return wordnet.synsets(word)[0].lemmas()[0].name()
    except IndexError:
        return word
