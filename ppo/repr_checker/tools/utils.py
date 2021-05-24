from typing import List


def recover_sentence(sents: List[str]) -> str:
    # Connect sentences into a complete text.
    # ex) f(['Hello World.', "It's good to see you.", 'Thanks for buying this book.'])
    # -> "Hello World. It's good to see you. Thanks for buying this book."
    return ' '.join(sents)
