from typing import List
import numpy as np

from .noisy import \
    shuffle_words, switch_words, confuse_tense_number, duplicate_words, replace_similar, \
    replace_random, insert_random, remove_word, replace_ing


_METHODS = [
    shuffle_words, switch_words, confuse_tense_number, duplicate_words, replace_similar,
    replace_random, insert_random, remove_word, replace_ing
]


def make_noisy(words: List[str], again: bool = False) -> List[str]:
    # Apply methods randomly
    before = words.copy()

    n = np.random.randint(1, 2)  # how many times to apply methods
    for _ in range(n):
        if len(words) < 3:
            return None

        i = np.random.randint(0, len(_METHODS))
        method = _METHODS[i]
        import time
        tmr = time.time()
        words = method(words)
        if time.time() - tmr > 0.1:
            print()
            print(time.time() - tmr, i)

        if words is None:
            words = before.copy()


    if words == before:
        # if there is no change,
        if again:
            # apply methods once again.
            return make_noisy(words, again=True)
        else:
            return None

    return words
