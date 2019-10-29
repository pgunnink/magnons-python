from magnons.process import Process
import os
from magnons.amplitudes import AkBkAngle
from magnons.energies import hamiltonian_AB
from timeit import timeit
import numpy as np
import scipy
if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    NN = 5
    p = Process(dir_path)
    r = p.get_all_kwargs()[-1]
    apply_kwargs, save_kwargs = p.create_save_and_run_kwargs(r)
    ky = 10**4
    kz = 10**4
    apply = {}
    for k in [
            'phi', 'alpha', 'N', 'J', 'S', 'h', 'eps', 'a', 'mu', 'Nr', 'Ng'
    ]:
        apply[k] = apply_kwargs[k]
    AkBkAngle(ky, kz, **apply)