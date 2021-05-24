from typing import List
import numpy as np
from pattern.en import conjugate, PRESENT, PAST, SG, PL

from ..utils import range_without, softmax, shuffle
from .nltk_utils import get_syn, RandomWord, to_ing

random_word = RandomWord()


# TODO: 소유격


def min_len_words(min_len: int):
    # Raise an error if text is too short.
    def decorator(fn):
        def inner(words: List[str], *args, **kwargs):
            assert len(words) >= min_len
            return fn(words, *args, **kwargs)
        return inner
    return decorator


@min_len_words(3)
def shuffle_words(words: List[str]) -> List[str]:
    # Shuffle few words
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['Hello', 'my', 'JunHee', 'is', 'name']
    two = min(max([2, len(words) // 2.5]), 3)
    if two == 2:
        n = 2
    else:
        n = np.random.randint(2, two+1)  # how many times to shuffle words

    # 1. pick up two indexes (a, b)
    try:
        if len(words)-1-n == 0:
            a = 0
        else:
            a = np.random.randint(0, len(words)-1-n)
    except ValueError:
        print(words, n)
        print(words, n)
        print(words, n)
        print(words, n)
        print(words, n)
        print(words, n)
        return words
    b = a + n
    # 2. shuffle them
    before = words.copy()
    cnt = 0
    while words == before:
        words = words[:a] + shuffle(words[a:b]) + words[b:]
        cnt += 1
        if cnt >= 10:
            break

    return words


@min_len_words(3)
def switch_words(words: List[str]) -> List[str]:
    # Pick up few words and switch them
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['name', 'my', 'Hello', 'is', 'JunHee']
    n = np.random.randint(1, min(max([1, len(words) // 3]), 3)+1)  # how many times to switch words

    for _ in range(n):
        # 1. pick up two words (index a, b)
        a = np.random.randint(0, len(words)-1)
        candidates = range_without(a, len(words))
        b = np.random.choice(candidates, p=softmax([-abs(a-i)**0.5 for i in candidates]))
        # 2. switch them
        tmp = words[b]
        words[b] = words[a]
        words[a] = tmp

    return words


@min_len_words(2)
def confuse_tense_number(words: List[str]) -> List[str]:
    # Change tense or number of a word
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['Hello', 'my', 'named', 'is', 'JunHee']
    candidates = [(PRESENT, SG), (PRESENT, PL), (PAST, SG), (PAST, PL)]
    n = np.random.randint(1, min(max([1, len(words) // 3]), 3)+1)  # how many times to switch words
    for _ in range(n):
        for _ in range(len(words)*2):
            i = np.random.randint(0, len(words)-1)  # index of word will be changed
            for cond in shuffle(candidates):
                try:
                    word = conjugate(words[i], tense=cond[0], number=cond[1])  # do
                except RuntimeError:
                    continue
                if word != words[i]:
                    break
            if word != words[i]:
                break
        words[i] = word
    return words


@min_len_words(1)
def duplicate_words(words: List[str]) -> List[str]:
    # Duplicate a word and insert it next to the word
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['Hello', 'my', 'name', 'is', 'is', 'JunHee']
    n = np.random.randint(1, min(max([2, len(words) // 3]), 4)+1)  # how many times to insert a duplicated word
    for _ in range(n):
        i = np.random.randint(0, len(words) - 1)  # index of word will be duplicated
        words.insert(i, words[i])
    return words


@min_len_words(1)
def replace_similar(words: List[str]) -> List[str]:
    # Replace a word with similar one (synonym).
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['Hello', 'my', 'citation', 'is', 'JunHee']
    n = np.random.randint(1, min(max([2, len(words) // 2]), 3)+1)  # how many times to replace
    for _ in range(n):
        for _ in range(len(words) * 2):
            i = np.random.randint(0, len(words) - 1)  # index of word will be replaced
            syn = get_syn(words[i])
            if syn is None:
                continue
            words[i] = syn
            break
    return words


@min_len_words(1)
def replace_random(words: List[str]) -> List[str]:
    # Replace a word with a random word.
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['Hello', 'situation', 'name', 'is', 'JunHee']
    n = np.random.randint(1, min(max([2, len(words) // 2]), 3)+1)  # how many times to replace
    for _ in range(n):
        i = np.random.randint(0, len(words) - 1)  # index of word will be replaced
        fn = np.random.choice([random_word.word, random_word.preposition,
                               random_word.stopword])
        word = fn()
        words[i] = word
    return words


@min_len_words(1)
def insert_random(words: List[str]) -> List[str]:
    # Insert a random word.
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['Hello', 'my', 'name', 'is', 'music', 'JunHee']
    n = np.random.randint(1, min(max([2, len(words) // 3]), 3)+1)  # how many times to insert
    for _ in range(n):
        i = np.random.randint(0, len(words))  # index to insert
        fn = np.random.choice([random_word.word, random_word.preposition,
                               random_word.article, random_word.stopword])
        word = fn()
        words.insert(i, word)
    return words


@min_len_words(2)
def remove_word(words: List[str]) -> List[str]:
    # Remove words randomly
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['Hello', 'my', 'is', 'JunHee']
    n = np.random.randint(1, min(max([2, len(words) // 3]), 3)+1)  # how many times to insert
    for _ in range(n):
        i = np.random.randint(0, len(words) - 1)  # index of word will be removed
        del words[i]
    return words


@min_len_words(1)
def replace_ing(words: List[str]) -> List[str]:
    # Change a word to a gerund
    # ex) f(['Hello', 'my', 'name', 'is', 'JunHee']) -> ['Hello', 'my', 'name', 'being', 'JunHee']
    for i in shuffle(list(range(len(words)))):
        ing = to_ing(words[i])
        if ing is not None and ing != words[i]:
            words[i] = ing
            return words


