import numpy as np


class Buffer:
    def __init__(self, maxlen: int = 100):
        self.maxlen = maxlen
        self.buffer = []

    def append(self, x):
        self.buffer.append(x)
        self.check_length()

    def check_length(self):
        while self.maxlen < len(self.buffer):
            del self.buffer[0]

    def mean(self):
        return sum(self.buffer) / len(self.buffer)

    def percentile(self, p: int):
        # return percentile
        # p : 0~100 int
        # if p==50 -> mean
        return np.percentile(self.buffer, p)
