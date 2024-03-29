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
from scipy import linalg


def get_energies_angle(k,
                       phi,
                       alpha,
                       E_to_GHz=None,
                       return_eigenfunctions=False,
                       N=None,
                       J=None,
                       S=None,
                       h=None,
                       eps=None,
                       a=None,
                       mu=None,
                       Nr=4,
                       Ng=4):
    ky = k[0]
    kz = k[1]
    A, B = AkBkAngle(ky,
                     kz,
                     phi,
                     alpha,
                     N=N,
                     J=J,
                     S=S,
                     h=h,
                     eps=eps,
                     a=a,
                     mu=mu,
                     Nr=Nr,
                     Ng=Ng)
    return get_E_and_ev(A, B, return_eigenfunctions, E_to_GHz)


def get_energies(k,
                 E_to_GHz=None,
                 return_eigenfunctions=False,
                 alpha=None,
                 N=None,
                 J=None,
                 S=None,
                 h=None,
                 eps=None,
                 a=None,
                 mu=None,
                 Nr=4,
                 Ng=4):
    ky = k[0]
    kz = k[1]
    A, B = AkBk(ky, kz, N=N, J=J, S=S, h=h, eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    return get_E_and_ev(A, B, return_eigenfunctions, E_to_GHz)


def get_E_and_ev(A, B, return_eigenfunctions, E_to_GHz):
    D = np.block([[A, B], [B.T, A.T]])
    N = A.shape[0]
    K = linalg.cholesky(D, overwrite_a=True, check_finite=False, lower=False)
    Ip = np.block([[np.identity(N), np.zeros((N, N))],
                   [np.zeros((N, N)), -np.identity(N)]])
    M = K @ Ip @ K.conj().T
    if return_eigenfunctions:
        E, U = linalg.eigh(M, overwrite_a=True, check_finite=False)
        E = Ip @ E  # note that linalg.eigh returns sorted eigenvalues
        E = np.abs(E)  # we want only positive values anyway
        inverse_K = K.T.conj()
        ev = inverse_K @ U @ np.diag(E**(1 / 2))

        E = E[:N] * E_to_GHz
        ev = ev[:, :N]
        # normalize:
        ev = ev * np.sqrt(np.sum(np.abs(ev)**2, axis=0)[np.newaxis, :])**(-1)
        E = np.flip(E)  # sorted from lowest to highest
        # same sorting as eigenvalues,
        # plus flipping the first axis to correspond to the old calculation:
        ev = np.flip(ev)
        return E, ev
    else:
        E = linalg.eigh(M,
                        eigvals_only=True,
                        overwrite_a=True,
                        check_finite=False)
        E = Ip @ E
        return np.abs(E)[:N] * E_to_GHz, None


def get_E_and_ev_old(A, B, return_eigenfunctions, E_to_GHz):
    mat = hamiltonian_AB(A, B)
    N = int(len(mat) / 2)

    if return_eigenfunctions:
        E, ev = linalg.eig(mat, check_finite=False)
        E = np.real(E)
        idx = np.argsort(E)
        E = E[idx]
        ev = ev[:, idx]
        E = E[-N:] * E_to_GHz
        ev = ev[:, -N:]

    else:
        eig = np.real(linalg.eigvals(mat))
        eig = np.sort(eig)
        E = eig[-N:] * E_to_GHz
        ev = None
    return E, ev


def klist_ev_in_HP_basis(ev):
    res = []
    for x in ev:
        res.append(ev_in_HP_basis(x))
    return np.array(res)


def ev_HP_to_S(ev, S=1):
    Sx = np.sqrt(2 * S) / 2 * (ev + ev.conj())
    Sy = np.sqrt(2 * S) / 2j * (ev - ev.conj())
    Sz = S - ev * ev.conj()
    return np.stack((Sx, Sy, Sz))


def ev_in_HP_basis(ev):
    if len(ev.shape) == 1:
        N = int(ev.shape[0] / 2)
        return ev[:N] + ev[N:].conj()
    else:
        N = ev.shape[1]
        res = np.zeros((N, N), dtype=np.complex)
        for i in range(N):
            res[:, i] = ev[:N, i] + ev[N:, i].conj()
        return res


def hamiltonian_AB(A, B):
    return np.block([[A, B], [-B.conj().T, -A.T]])


def plot_eigenfunction(k, n, kvalues, E, ev):
    diff = np.abs(kvalues - np.array(k))
    diff_distance = np.sqrt(np.sum(diff**2, axis=1))
    i = np.argmin(diff_distance)
    ev = ev[i]
    E = E[i]
    ev = ev_in_HP_basis(ev)
    return plt.plot(np.abs(ev[:, n])**2)


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
    return np.sqrt((h + rho * absk**2 + Delta * (1 - fk) * np.sin(theta)**2) *
                   (h + rho * absk**2 + Delta * fk)) * E_to_GHz


def get_dispersion_theta(theta,
                         Nk,
                         phi=0,
                         alpha=0,
                         use_angled_if_zero=False,
                         ky_begin=2,
                         ky_end=6,
                         logspace=True,
                         parallel=True,
                         return_eigenfunctions=False,
                         N=100,
                         J=None,
                         S=None,
                         h=None,
                         eps=None,
                         a=None,
                         mu=None,
                         Nr=4,
                         Ng=4):
    if logspace:
        kvalues = np.logspace(ky_begin, ky_end, Nk)
    else:
        kvalues = np.linspace(10**ky_begin, 10**ky_end, Nk)
    ky = kvalues * np.sin(theta)  #+ 10**-20
    kz = kvalues * np.cos(theta)  #+ 10**-20
    kvalues = np.stack((ky, kz), axis=1)

    if phi != 0 or (phi == 0 and use_angled_if_zero):
        f = partial(get_energies_angle,
                    phi=phi,
                    alpha=alpha,
                    return_eigenfunctions=return_eigenfunctions,
                    N=N,
                    J=J,
                    S=S,
                    h=h,
                    eps=eps,
                    a=a,
                    mu=mu,
                    Nr=Nr,
                    Ng=Ng,
                    E_to_GHz=E_to_GHz)
    else:
        f = partial(get_energies,
                    return_eigenfunctions=return_eigenfunctions,
                    N=N,
                    J=J,
                    S=S,
                    h=h,
                    eps=eps,
                    a=a,
                    mu=mu,
                    Nr=Nr,
                    Ng=Ng,
                    E_to_GHz=E_to_GHz)
    if parallel:
        with Pool(4) as p:
            energies = np.zeros((Nk, N))
            eigenfunctions = np.zeros((Nk, N * 2, N), dtype=np.complex)
            for i, (E, ev) in enumerate(tqdm(p.imap(f, kvalues), total=Nk)):
                energies[i, :] = E
                eigenfunctions[i, :] = ev
    else:
        energies = np.zeros((Nk, N))
        eigenfunctions = np.zeros((Nk, N * 2, N), dtype=np.complex)
        for i, (E, ev) in enumerate(tqdm(map(f, kvalues), total=Nk)):
            energies[i, :] = E
            eigenfunctions[i, :] = ev
    return np.array(energies), np.array(eigenfunctions), np.array(kvalues)


def plot_dispersion_ky(res, kvalues):
    plt.figure()
    for i in range(6):
        plt.semilogx(kvalues, res[:, i], "-.", color="black")
    plt.show()


if __name__ == "__main__":
    get_energies_angle([10**(-3), 10**3],
                       np.radians(40),
                       np.radians(60),
                       E_to_GHz=E_to_GHz,
                       return_eigenfunctions=True,
                       N=400,
                       J=J,
                       S=S,
                       h=2500 * mu,
                       mu=mu,
                       eps=a**(-2),
                       a=a)
