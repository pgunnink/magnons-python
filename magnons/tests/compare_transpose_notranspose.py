import matplotlib.pyplot as plt
import numpy as np
from magnons.yig import a, S, mu, J, Ms
from magnons.cgs import E_to_GHz
from magnons.energies import get_energies, energies_uniform_mode, get_dispersion_theta
from tqdm import tqdm
from copy import copy
from numpy import pi
from magnons.amplitudes import AkBk
from multiprocessing import Pool
from functools import partial
import os


def f_transpose(k,
                N=None,
                J=None,
                S=None,
                h=None,
                eps=None,
                a=None,
                mu=None,
                E_to_GHz=None):
    ky = k[0]
    kz = k[1]
    A, B = AkBk(ky, kz, N=N, a=a, J=J, S=S, h=h, eps=eps, mu=mu)
    mat = np.block([[A, B], [-B.conj().T, -A]])
    eig = np.real(np.linalg.eigvals(mat))
    eig = np.sort(eig)
    return eig[eig > 0] * E_to_GHz


def f_notranspose(k,
                  N=None,
                  J=None,
                  S=None,
                  h=None,
                  eps=None,
                  a=None,
                  mu=None,
                  E_to_GHz=None):
    ky = k[0]
    kz = k[1]
    A, B = AkBk(ky, kz, N=N, a=a, J=J, S=S, h=h, eps=eps, mu=mu)
    mat = np.block([[A, B], [-B.conj(), -A.T]])
    eig = np.real(np.linalg.eigvals(mat))
    eig = np.sort(eig)
    return eig[eig > 0] * E_to_GHz


if __name__ == '__main__':
    N = 400
    H = 700
    kwargs = {
        "eps": a**-2,
        "a": a,
        "S": S,
        "mu": mu,
        "J": J,
        "E_to_GHz": E_to_GHz,
        "h": mu * H,
    }
    kwargs["N"] = N
    kwargs_uni = {
        "a": a,
        "mu": mu,
        "M": Ms,
        "E_to_GHz": E_to_GHz,
        "rho": J * S * a**2,
        "N": N,
        "h": mu * H,
    }
    E0 = np.sqrt(mu * H * (mu * H + 4 * pi * mu * Ms)) * E_to_GHz
    Nk = 20
    directory = './tests/transpose_notranspose'
    os.makedirs(directory, exist_ok=True)

    for theta in tqdm([0, 30, 60, 90]):
        kvalues = np.logspace(4, 6, Nk)
        ky = kvalues * np.cos(np.radians(theta)) + 10**(-20)
        kz = kvalues * np.sin(np.radians(theta)) + 10**(-20)
        kvalues = np.stack((ky, kz), axis=1)
        f_transpose_lam = partial(f_transpose, **kwargs)
        f_notranspose_lam = partial(f_notranspose, **kwargs)
        with Pool(4) as p:
            res_transpose = []
            res_notranspose = []
            for x in tqdm(p.imap(f_transpose_lam, kvalues), total=Nk):
                res_transpose.append(x[:6])
            for x in tqdm(p.imap(f_notranspose_lam, kvalues), total=Nk):
                res_notranspose.append(x[:6])
        # res_transpose = []
        # for x in map(f_transpose_lam, kvalues):
        #     res_transpose.append(x[:6])
        res_transpose = np.array(res_transpose)
        res_notranspose = np.array(res_notranspose)

        plt.figure()
        absk = np.sqrt(np.sum(kvalues**2, axis=1))
        for i in range(6):
            plt.semilogx(absk, res_transpose[:, i], "-", color="black")

        uni = [energies_uniform_mode(*x, **kwargs_uni) for x in kvalues]
        plt.semilogx(absk, uni, "r--")
        plt.axhline(E0)
        minimum = res_transpose.min()
        plt.ylim(top=5.5, bottom=minimum * 0.95)
        plt.xlabel(r"$k\ (cm^{-1})$")
        plt.ylabel(r"$E_k\ (GHz)$")

        plt.savefig(
            f"./tests/transpose_notranspose/theta_{theta}_withtranspose.png")

        plt.figure()
        for i in range(6):
            plt.semilogx(absk, res_notranspose[:, i], "-", color="black")

        uni = [energies_uniform_mode(*x, **kwargs_uni) for x in kvalues]
        plt.semilogx(absk, uni, "r--")
        plt.axhline(E0)
        minimum = res_notranspose.min()
        plt.ylim(top=5.5, bottom=minimum * 0.95)
        plt.xlabel(r"$k\ (cm^{-1})$")
        plt.ylabel(r"$E_k\ (GHz)$")
        plt.savefig(
            f"./tests/transpose_notranspose/theta_{theta}_notranspose.png")
