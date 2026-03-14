import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    N = len(seqs)
    L = max_len

    result = np.full((N, L), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        length = min(len(seq), L)
        result[i, :length] = seq[:length]

    return result