import random
from typing import List, Dict
from sortedcontainers import SortedSet

from correlationlength.util import split_into_sentences, split
from correlationlength import configs


def load_cat2probes(probes_names: List[str],
                    ) -> Dict[str, SortedSet]:
    res = {}
    for pn in probes_names:
        p = configs.Dirs.probes / f'{pn}.txt'
        text_in_file = p.read_text()
        probes = text_in_file.split('\n')
        probes.remove('')
        res[pn] = SortedSet(probes)
    return res


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
    shuffling the documents does not remove all age-structure,
    because utterances associated with teh same age are still clustered within documents.
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
        tokens_new = [t for sentence in sentences for t in sentence]
        num_original_docs = len(text_in_file.split('\n'))
        size = len(tokens_new) // num_original_docs
        docs = [' '.join(tokens) for tokens in split(tokens_new, size)]  # convert back to strings
    else:
        docs = text_in_file.split('\n')

    num_docs = len(docs)
    print(f'Loaded {num_docs} documents from {corpus_name}')

    return docs
