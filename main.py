"""
Estimate correlation length of a corpus of text, one for each supplied set of probe words.

This is useful, for example, when deciding on the number of backpropagation-through-time steps
 when training a RNN language model.

Notes:
    By default, sentences are shuffled,
     so that correlation length excludes topical dependencies between words of different sentences in the same document.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from preppy import Prep as TrainPrep
from preppy.util import make_windows_mat

from correlationlength.io import load_docs, load_cat2probes
from correlationlength.util import calc_kl_divergence


CORPUS_NAME = 'childes-20201026'
PROBES_NAME = 'nouns-2972'
CONTEXT_SIZE = 8
NUM_PARTS = 2  # must be 1 to include all windows in windows matrix
SHUFFLE_SENTENCES = True  # if not True, document-level dependencies result in very long distance dependencies

# load corpus
docs = load_docs(CORPUS_NAME, shuffle_sentences=SHUFFLE_SENTENCES)
prep = TrainPrep(docs,
                 reverse=False,
                 sliding=False,
                 num_parts=NUM_PARTS,
                 num_iterations=(1,1),
                 batch_size=1,
                 context_size=CONTEXT_SIZE,
                 )

# load probes
probes_ = load_cat2probes(PROBES_NAME)

part2y = defaultdict(list)
for part_id, token_ids in enumerate(prep.reordered_parts):
    print(f'Partition {part_id + 1}')

    windows_mat = make_windows_mat(token_ids, prep.num_windows_in_part, prep.num_tokens_in_window)

    # exclude any probes not in partition
    probes = []
    for p in probes_:
        if p not in prep.token2id:
            print(f'WARNING: Excluding "{p}".')
        else:
            probes.append(p)

    # condition on a subset of words
    cat_probe_ids = [prep.token2id[p] for p in probes]
    bool_ids = np.isin(windows_mat[:, -1], cat_probe_ids)

    # un-conditional probability of words in the whole text
    token_id_types, token_id_counts = np.unique(windows_mat, return_counts=True)
    unconditional_probabilities = token_id_counts / np.sum(token_id_counts)

    # kl divergence due to chance
    col_chance = np.random.choice(token_ids,  # samples token ids from whole corpus (frequency-sensitive)
                                  size=np.sum(bool_ids), replace=True)
    outcomes, c = np.unique(col_chance, return_counts=True)
    o2p = dict(zip(outcomes, c / np.sum(c)))
    q_chance = np.array([o2p.setdefault(o, 0) for o in token_id_types])
    kl_chance = calc_kl_divergence(unconditional_probabilities, q_chance)

    for n, col in enumerate(windows_mat[bool_ids].T):

        # actual kl divergence
        outcomes, c = np.unique(col, return_counts=True)
        o2p = dict(zip(outcomes, c / np.sum(c)))
        q = np.array([o2p.setdefault(o, 0) for o in token_id_types])
        kl = calc_kl_divergence(unconditional_probabilities, q)

        print(f'distance={CONTEXT_SIZE - n:>2} kl={kl:.3f} {"<" if kl <= kl_chance else ">"} {kl_chance:.3f}')

        part2y[part_id + 1].append(kl)

    print()


# fig
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
plt.title(f'Correlation Length\nCorpus={CORPUS_NAME}')
ax.set_ylabel('KL Divergence', fontsize=12)
ax.set_xlabel(f'Distance from target ({PROBES_NAME})', fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
for part, y in part2y.items():
    y = np.flip(y)[1:]
    x = np.arange(1, len(y) + 1)
    ax.plot(x, y, '-', label=f'Corpus Partition {part}')
plt.yscale('log')
plt.legend(frameon=False)
plt.show()
