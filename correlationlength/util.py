import numpy as np
from typing import List, Set


def split(l, split_size):
    for i in range(0, len(l), split_size):
        yield l[i:i + split_size]


def split_into_sentences(tokens: List[str],
                         punctuation: Set[str],
                         ) -> List[List[str]]:
    assert isinstance(punctuation, set)

    res = [[]]
    for w in tokens:
        res[-1].append(w)
        if w in punctuation:
            res.append([])
    return res


def calc_kl_divergence(p: np.ndarray,
                       q: np.ndarray,
                       epsilon=0.00001,
                       ) -> float:
    assert len(p) == len(q)

    pe = p + epsilon
    qe = q + epsilon
    res = np.sum(pe * np.log2(pe / qe)).item()
    return res
