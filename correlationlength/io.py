import random
from typing import List, Dict
from sortedcontainers import SortedSet

from correlationlength.util import split_into_sentences, split
from correlationlength import configs


def load_cat2probes(probes_name: str,
                    lower_case: bool = True,
                    ) -> SortedSet:

    p = configs.Dirs.probes / f'{probes_name}.txt'
    text_in_file = p.read_text()
    probes = [p.lower() if lower_case else p for p in text_in_file.split('\n')]
    probes.remove('')
    return SortedSet(probes)


def load_docs(corpus_name: str,
              shuffle_sentences: bool = False,
              shuffle_seed: int = 20,
              ) -> List[str]:

    """
    A "document" has type string. It is not tokenized.

    WARNING:
    Always use a seed for random operations.
    For example when loading tags and words using this function twice, they won't align if no seed is set

    WARNING:
    shuffling is at the sentence-level rather than the document-level,
    because shuffling at document-level does not remove all age-structure.
    Utterances associated with the same age are still clustered within documents.
    """

    p = configs.Dirs.corpora / f'{corpus_name}.txt'
    text_in_file = p.read_text()

    # shuffle at sentence-level (as opposed to document-level)
    # this remove clustering of same-age utterances within documents
    if shuffle_sentences:
        random.seed(shuffle_seed)
        print('WARNING: Shuffling sentences')
        tokens = text_in_file.replace('\n', ' ').split()
        sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
        random.shuffle(sentences)
        tokens = [t for sentence in sentences for t in sentence]
    else:
        tokens = text_in_file.replace('\n', ' ').split()

    docs = text_in_file.split('\n')
    num_docs = len(docs)
    num_tokens = len(tokens)
    print(f'Loaded {num_docs:,} documents with {num_tokens:,} tokens documents from {corpus_name}')

    return tokens
