import numpy as np


def get_spincurrent(ev):
    current = np.zeros(len(ev), dtype=np.complex)
    if len(ev.shape) > 1:
        raise Exception('only compatible with shape (N,)')
    ev = np.pad(ev, 1, 'constant', constant_values=0)
    for i in range(1, len(ev) - 1):
        current[i - 1] = ev[i].conj() * ev[i + 1] - ev[i].conj() * ev[i - 1]
    return 1j * current
