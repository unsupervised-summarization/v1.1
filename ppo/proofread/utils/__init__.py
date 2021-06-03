from typing import List
import numpy as np
from .logger import Logger


def range_without(without: int, a: int, b: int = None, c: int = None) -> List[int]:
    # return range(a, b, c) without `without`
    # ex) f(2, 0, 5) -> [0, 1, 3, 4]
    if b is None:
        r = list(range(a))
    elif c is None:
        r = list(range(a, b))
    else:
        r = list(range(a, b, c))
    r.remove(without)
    return r


def softmax(a: list) -> np.ndarray:
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def shuffle(x: list) -> list:
    np.random.shuffle(x)
    return x
