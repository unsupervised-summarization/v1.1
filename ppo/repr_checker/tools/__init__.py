from typing import List
import numpy as np

from .utils import recover_sentence


def extract_sent(sents: List[str], ignore_short: bool = False) -> (str, str):
    # Randomly extract a sentence from `sents` (sequence of sentences).
    # return (text from `sents`, an extracted sentence)

    if len(sents) <= 2 and not ignore_short:
        return None, None

    if np.random.random() >= 0.5:
        n = np.random.randint(1, min(len(sents) // 2 + 1 + int(ignore_short), 3))  # how many sentences to extract
    else:
        n = 1  # how many sentences to extract

    a = np.random.randint(0, len(sents)-n+1)
    b = a + n
    extracted: List[str] = sents[a:b]
    now_sents: List[str] = sents[:a] + sents[b:]

    extracted_sentence: str = recover_sentence(extracted)
    now_text: str = recover_sentence(now_sents)

    return now_text, extracted_sentence
