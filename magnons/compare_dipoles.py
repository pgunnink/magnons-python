from magnons.dipolar_sums import Dkxx, Dkyy, Dkxy
from numpy import pi
import numpy as np
from magnons.yig import mu, a
import matplotlib.pyplot as plt
from functools import partial


def Dkxx_uni(k, x, theta=None, mu=None, a=None):
    return 2 * pi * mu**2 / (a**2) * np.abs(k) * np.exp(-np.abs(k) * np.abs(x))


def Dkyy_uni(k, x, theta=None, mu=None, a=None):
    return -np.sin(theta)**2 * Dkxx_uni(k, x, mu=mu, a=a)


def Dkzz_uni(k, x, theta=None, mu=None, a=None):
    return -np.sin(theta)**2 * Dkxx_uni(k, x, mu=mu, a=a)


def Dkxy_uni(k, x, theta=None, mu=None, a=None):
    return 1j * 2 * np.pi * np.sin(theta) * mu**2 / (
        a**2) * np.abs(k) * x / np.abs(x) * np.exp(-np.abs(k) * np.abs(x))


if __name__ == "__main__":
    H = 700.0
    h = mu * H
    eps = a**(-2)

    K = Dkxx(eps, a, mu, Nr=10, Ng=10)
    funs = [
        Dkxx(eps, a, mu, Nr=10, Ng=10),
        Dkyy(eps, a, mu, Nr=10, Ng=10),
        Dkxy(eps, a, mu, Nr=10, Ng=10)
    ]
    uni_funs = [
        partial(Dkxx_uni, mu=mu, a=a),
        partial(Dkyy_uni, mu=mu, a=a),
        partial(Dkxy_uni, mu=mu, a=a)
    ]
    parse_funs = [np.real, np.real, np.imag]
    titles = ['Dkxx', 'Dkyy', 'Dkxy']
    for K, f, parse, tit in zip(funs, uni_funs, parse_funs, titles):
        plt.figure()
        plt.suptitle(tit)
        for i, ky in enumerate(np.logspace(3, 6, 4)):
            kz = 10**4
            theta = np.arctan2(ky, kz)
            absk = np.sqrt(ky**2 + kz**2)
            lim = 20
            xrange = np.arange(1, lim) * a
            plt.subplot(2, 2, i + 1)
            plt.title(f"k = 10^{np.log10(ky)}")
            xlinspace = np.linspace(0.1, 20, 100) * a
            plt.plot(
                xlinspace / a,
                [parse(f(absk, x, theta=theta)) for x in xlinspace],
                "-",
                label="Uniform mode",
            )
            plt.plot(xrange / a, [parse(K.run(x, ky, kz)) for x in xrange],
                     "*",
                     label="Ewald")
            plt.legend()

    plt.show()
