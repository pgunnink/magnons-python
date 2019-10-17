from magnons.dipolar_sums import Dkxx, Dkyy
from numpy import pi
import numpy as np
from magnons.yig import mu, a
import matplotlib.pyplot as plt


def Dkxx_uni(k, x, mu=None, a=None):
    return 2 * pi * mu**2 / (a**2) * k * np.exp(-k * x)


if __name__ == "__main__":
    H = 900.0
    h = mu * H
    eps = a**(-2)
    K = Dkyy(eps, a, mu, Nr=10, Ng=10)
    for i, ky in enumerate(np.logspace(3, 6, 4)):
        kz = 10**4
        theta = np.arctan2(ky, kz)
        absk = np.sqrt(ky**2 + kz**2)
        xrange = np.arange(1, 100) * a
        plt.subplot(2, 2, i + 1)
        plt.title(np.log10(ky))
        xlinspace = np.linspace(0.1, 100, 100) * a
        plt.plot(
            xlinspace / a,
            [
                Dkxx_uni(absk, x, mu=mu, a=a) * -np.sin(theta)**2
                for x in xlinspace
            ],
            "-",
            label="Uniform mode",
        )
        plt.plot(xrange / a, [K.run(x, ky, kz) for x in xrange],
                 "*",
                 label="Ewald")
        plt.legend()

    plt.show()
