from magnons.dipolar_sums import Dkxx
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt


def Dkxx_uni(k, x, mu=None, a=None):
    return 2 * pi * mu**2 / (a**2) * k * np.exp(-k * x)


def Dkxx_uni_fk(k, mu=None, a=None, d=None):
    return 4 * pi * mu**2 / (a**3) * (1 / 3 - (1 - np.exp(-k * d)) / (k * d))


if __name__ == "__main__":
    pass
