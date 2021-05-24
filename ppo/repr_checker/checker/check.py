import numpy as np
from numpy.linalg import norm
import hashlib
import torch

from ..model import get_feature


class ReprChecker:
    def __init__(self, max_cache_len=100):
        self.cache = {}
        self.max_cache_len = max_cache_len

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
        if caching:
            # return cached result
            document_feature: np.ndarray = self.get_cache(document.lower(), else_=get_feature)
            summary_feature: np.ndarray = self.get_cache(summary.lower(), else_=get_feature)
        else:
            document_feature: np.ndarray = get_feature(document.lower())
            summary_feature: np.ndarray = get_feature(summary.lower())

        sim: float = self.cos_sim(document_feature, summary_feature)  # range 0 ~ 1
        return float(sim)

    @staticmethod
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        # Calculate a cosine similarity
        return float(1 - np.clip(np.mean((a-b) ** 2), 0, 1))

