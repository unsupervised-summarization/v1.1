import numpy as np
from numpy.linalg import norm
import hashlib
import torch

from ..data import DataLoader
from ..model import tokenizer, model


class ReprChecker:
    def __init__(self, max_cache_len=100):
        self.cache = {}
        self.max_cache_len = max_cache_len

    def load(self, path: str):
        model.load_state_dict(torch.load(path))

    def get_cache(self, x, else_=None):
        key = hashlib.md5(x.encode()).digest()
        if key in self.cache:
            return self.cache[key]
        else:
            if else_ is None:
                return None
            r = else_(x)
            while len(self.cache.keys()) >= self.max_cache_len:
                del self.cache[list(self.cache.keys())[0]]
            self.cache[key] = r
            return r

    def check(self, document: str, summary: str, caching: bool = False) -> float:
        # Return a semantic similarity (0~1) between a document and a summary.
        document = DataLoader.preprocessing(document)
        summary = DataLoader.preprocessing(summary)
        input_text = summary + ' [CLS] ' + document
        if caching:
            out = self.get_cache(input_text, else_=self.predict)
        else:
            out = self.predict(input_text)  # 0 ~ 1
        return out

    @staticmethod
    def predict(input_text: str) -> float:
        inputs = tokenizer(input_text, return_tensors='pt')
        outputs = model(**inputs)
        outputs = torch.sigmoid(outputs[0])
        out = float(outputs[0][0].cpu().detach().numpy())  # 0 ~ 1
        return out
