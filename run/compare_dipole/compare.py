from magnons.dipolar_sums import Dkxx
import numpy as np
from magnons.yig import mu, a
import matplotlib.pyplot as plt


def fk(k, d):
    return (1 - np.exp(-np.abs(k) * d)) / (np.abs(k) * d)


def Dkxx_integrated(k, d, mu, a):
    return 4 * np.pi * mu**2 / a**3 * (1 / 3 - fk(k, d))


if __name__ == '__main__':
    H = 700.0
    h = mu * H
    eps = a**(-2)
    N = 100
    K = Dkxx(eps, a, mu, Nr=10, Ng=10)
    ky = 10**3
    kz = 10**2
    kabs = np.sqrt(ky**2 + kz**2)
    r = K.table(ky, kz, N)
    d = len(r) * a
    res_exact = np.sum(r)
    res_approx = Dkxx_integrated(kabs, d, mu, a)
    print(f"Exact {res_exact}, approx: {res_approx}")
