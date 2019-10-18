from magnons import amplitudes
from magnons.amplitudes import AkBk, AkBkAngle
import numpy as np
from magnons.yig import a, S, mu, J
from magnons.cgs import E_to_GHz
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from numpy import pi


def energies_angle(k, phi, E_to_GHz=None, **kwargs):
    ky = k[0]
    kz = k[1]
    A, B = AkBkAngle(ky, kz, phi, **kwargs)
    mat = np.block([[A, B], [-B.conj().T, -A]])
    eig = np.real(np.linalg.eigvals(mat))
    eig = np.sort(eig)
    return eig[eig > 0] * E_to_GHz


def energies(k, E_to_GHz=None, **kwargs):
    ky = k[0]
    kz = k[1]
    A, B = AkBk(ky, kz, **kwargs)
    mat = np.block([[A, B], [-B.conj().T, -A]])
    eig = np.real(np.linalg.eigvals(mat))
    eig = np.sort(eig)
    return eig[eig > 0] * E_to_GHz


def energies_uniform_mode(ky,
                          kz,
                          h=None,
                          mu=None,
                          M=None,
                          rho=None,
                          N=None,
                          a=None,
                          E_to_GHz=None):
    theta = np.arctan2(ky, kz)
    absk = np.sqrt(ky**2 + kz**2)
    Delta = 4 * pi * mu * M
    d = N * a
    fk = (1 - np.exp(-absk * d)) / (absk * d)
    return (np.sqrt((h + rho * absk**2 + Delta * (1 - fk) * np.sin(theta)**2) *
                    (h + rho * absk**2 + Delta * fk)) * E_to_GHz)


def get_dispersion_theta(theta,
                         Nk,
                         phi=0,
                         alpha=0,
                         use_angled_if_zero=False,
                         ky_begin=2,
                         ky_end=6,
                         firstN=6,
                         logspace=True,
                         parallel=True,
                         **kwargs):
    if logspace:
        kvalues = np.logspace(ky_begin, ky_end, Nk)
    else:
        kvalues = np.linspace(ky_begin, ky_end, Nk)
    ky = kvalues * np.cos(theta) + 10**-20
    kz = kvalues * np.sin(theta) + 10**-20
    kvalues = np.stack((ky, kz), axis=1)
    if phi != 0 or (phi == 0 and use_angled_if_zero):
        kwargs['phi'] = phi
        kwargs['alpha'] = alpha
        f = partial(energies_angle, **kwargs)
    else:
        f = partial(energies, **kwargs)
    if parallel:
        with Pool(4) as p:
            res = []
            for x in tqdm(p.imap(f, kvalues), total=Nk):
                res.append(x[:firstN])
    else:
        res = [f(x)[:firstN] for x in tqdm(kvalues, total=Nk)]
    return np.array(res), kvalues


def plot_dispersion_ky(res, kvalues):
    plt.figure()
    for i in range(6):
        plt.semilogx(kvalues, res[:, i], "-.", color="black")
    plt.show()


if __name__ == "__main__":
    kwargs = {
        "eps": a**-2,
        "a": a,
        "S": S,
        "mu": mu,
        "J": J,
        "E_to_GHz": E_to_GHz,
        "h": mu * 700,
        "Nr": 4,
        "Ng": 4,
    }
    kwargs["N"] = 400
