from numpy import pi
import numpy as np
from magnons.energies import hamiltonian_AB, get_E_and_ev
from magnons.dipolar_sums import Dkxx, Dkyy, Dkzz, Dkxy
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool


def Dkxx_uni(ky, kz, x, mu=None, a=None):
    k = np.sqrt(ky**2 + kz**2)
    if x == 0:
        return Dkxx(eps=a**(-2), a=a, mu=mu).run(0, ky, kz)
    return 2 * pi * mu**2 / (a**2) * np.abs(k) * np.exp(-np.abs(k) * np.abs(x))


def Dkyy_uni(ky, kz, x, theta=None, mu=None, a=None):
    k = np.sqrt(ky**2 + kz**2)
    if x == 0:
        return Dkyy(eps=a**(-2), a=a, mu=mu).run(0, ky, kz)
    return -1 * np.sin(theta)**2 * Dkxx_uni(ky, kz, x, mu=mu, a=a)


def Dkzz_uni(ky, kz, x, theta=None, mu=None, a=None):
    k = np.sqrt(ky**2 + kz**2)
    if x == 0:
        return Dkzz(eps=a**(-2), a=a, mu=mu).run(0, ky, kz)
    return -1 * np.cos(theta)**2 * Dkxx_uni(ky, kz, x, mu=mu, a=a)


def Dkxy_uni(ky, kz, x, theta=None, mu=None, a=None):
    k = np.sqrt(ky**2 + kz**2)
    if x == 0:
        return Dkxy(eps=a**(-2), a=a, mu=mu).run(0, ky, kz)
    return 1j * 2 * np.pi * np.sin(theta) * mu**2 / (
        a**2) * np.abs(k) * x / np.abs(x) * np.exp(-np.abs(k) * np.abs(x))


def AkBk(ky,
         kz,
         N=None,
         J=None,
         S=None,
         h=None,
         eps=None,
         a=None,
         mu=None,
         Nr=4,
         Ng=4):
    theta = np.arctan2(ky, kz)
    kabs = np.sqrt(ky**2 + kz**2)
    xx_table = np.array(
        [Dkxx_uni(ky, kz, x * a, mu=mu, a=a) for x in np.arange(-N + 1, N, 1)])
    yy_table = np.array([
        Dkyy_uni(ky, kz, x * a, theta=theta, mu=mu, a=a)
        for x in np.arange(-N + 1, N, 1)
    ])
    zz_table0 = np.array([
        Dkzz_uni(10**(-6), 10**(-6), x * a, theta=theta, mu=mu, a=a)
        for x in np.arange(-N + 1, N, 1)
    ])
    xy_table = np.array([
        Dkxy_uni(ky, kz, x * a, theta=theta, mu=mu, a=a)
        for x in np.arange(-N + 1, N, 1)
    ])

    Atemp = np.zeros((N, N), dtype=np.complex)
    Btemp = np.zeros((N, N), dtype=np.complex)
    Atemp += np.diag([h + S * np.sum(zz_table0[i:i + N]) for i in range(N)])
    Atemp += np.diag(
        np.ones(N) * S * J * (6 - 2 * np.cos(ky * a) - 2 * np.cos(kz * a)))
    Atemp[0, 0] -= S * J
    Atemp[N - 1, N - 1] -= S * J
    Atemp += np.diag(np.ones(N - 1), -1) * -J * S
    Atemp += np.diag(np.ones(N - 1), 1) * -J * S
    for i in range(N):
        Atemp[i, :] -= .5 * S * np.flip(xx_table[i:i + N] + yy_table[i:i + N])
        Btemp[i, :] -= .5 * S * np.flip(xx_table[i:i + N]
                                        - 2j * xy_table[i:i + N]
                                        - yy_table[i:i + N])
    return Atemp, Btemp


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
    mat = hamiltonian_AB(A, B)
    return get_E_and_ev(return_eigenfunctions, mat, E_to_GHz)


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
                         E_to_GHz=None,
                         Nr=4,
                         Ng=4):
    if logspace:
        kvalues = np.logspace(ky_begin, ky_end, Nk)
    else:
        kvalues = np.linspace(10**ky_begin, 10**ky_end, Nk)
    ky = kvalues * np.sin(theta) + 10**-20
    kz = kvalues * np.cos(theta) + 10**-20
    kvalues = np.stack((ky, kz), axis=1)

    if phi != 0 or (phi == 0 and use_angled_if_zero):
        pass
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
    energies = np.zeros((Nk, N))
    eigenfunctions = np.zeros((Nk, N * 2, N), dtype=np.complex)
    with Pool(4) as p:
        energies = np.zeros((Nk, N))
        eigenfunctions = np.zeros((Nk, N * 2, N), dtype=np.complex)
        for i, (E, ev) in enumerate(tqdm(p.imap(f, kvalues), total=Nk)):
            energies[i, :] = E
            eigenfunctions[i, :] = ev
    # for i, (E, ev) in enumerate(tqdm(map(f, kvalues), total=Nk)):
    #     energies[i, :] = E
    #     eigenfunctions[i, :] = ev
    return np.array(energies), np.array(eigenfunctions), np.array(kvalues)


if __name__ == '__main__':
    from magnons.yig import a, S, mu, J
    from magnons.cgs import E_to_GHz
    import matplotlib.pyplot as plt
    print(Dkxx(eps=a**(-2), a=a, mu=mu).table(10**2, 10**2, 10).shape)
    print(np.arange(-10 + 1, 10 + 2).shape)
    E, ev, k = get_dispersion_theta(np.radians(90),
                                    Nk=16,
                                    N=400,
                                    J=J,
                                    S=S,
                                    h=mu * 700,
                                    mu=mu,
                                    eps=a**(-2),
                                    a=a,
                                    E_to_GHz=E_to_GHz)
    kabs = np.sqrt(np.sum(k**2, axis=1))
    for i in range(6):
        plt.semilogx(kabs, E[:, i])
    plt.ylim(np.min(E) * .95, 5.5)
    plt.show()
