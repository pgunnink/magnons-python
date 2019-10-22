from magnons.energies import get_dispersion_theta
import numpy as np
from magnons.yig import a, S, mu, J, Ms
from magnons.cgs import E_to_GHz
import matplotlib.pyplot as plt
from magnons.magnetization import magnetization_angle
from magnons.data import Data
from copy import copy
import os
if __name__ == '__main__':
    N = 400
    H = 2500
    kwargs = {
        "eps": a**-2,
        "a": a,
        "S": S,
        "mu": mu,
        "J": J,
        "E_to_GHz": E_to_GHz,
        "h": mu * H,
        "Nr": 4,
        "Ng": 4,
    }
    theta = 0
    save_dir = './dispersion/canted_field/'
    os.makedirs(save_dir, exist_ok=True)
    save_dir_h5 = './data/'
    os.makedirs(save_dir_h5, exist_ok=True)
    save_file = save_dir_h5 + 'main_data.h5'
    kwargs["N"] = N
    firstN = 10
    alpha = [0, 30, 60, 80]
    plt.figure(dpi=200, figsize=(16, 8))
    for i, al in enumerate(alpha):
        al = np.radians(al)
        phi = magnetization_angle(al, M=Ms, H=H)
        res, ev, kvalues = get_dispersion_theta(theta,
                                                48,
                                                ky_begin=10**2,
                                                ky_end=2 * 10**6,
                                                phi=phi,
                                                firstN=firstN,
                                                use_angled_if_zero=True,
                                                logspace=False,
                                                return_eigenfunctions=True,
                                                **kwargs)

        absk = np.sqrt(np.sum(kvalues**2, axis=1))
        with Data(save_file) as f:
            save_kwargs = copy(kwargs)
            del save_kwargs["h"]
            save_kwargs["H"] = H
            save_kwargs["alpha"] = np.degrees(al)
            save_kwargs["phi"] = np.degrees(phi)
            save_kwargs["theta"] = np.degrees(theta)
            f.save_data(kvalues, res, ev, **save_kwargs)
        plt.subplot(2, 2, i + 1)
        for i in range(firstN):
            plt.plot(absk, res[:, i], "-", color="black")
        plt.title(
            f'H angle {np.round(np.degrees(al))}, M angle {np.round(np.degrees(phi))}'
        )
    plt.savefig(save_dir + 'canted_magnetic_field.png')
