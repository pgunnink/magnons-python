from scipy.optimize import fsolve, minimize, brent
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt


def magnetization_angle(alpha, M=None, H=None):
    """
    alpha: angle of magnetic field
    """
    if alpha == 0:
        return 0

    def f(x):
        return -M * H * np.cos(alpha - x) - 2 * pi * M**2 * np.cos(x)**2

    # phi, d, _, _ = fsolve(f, alpha, full_output=True, xtol=10**-5)
    phi = brent(f, brack=(0, alpha, 0.5 * pi))
    # if phi < 0:
    #     phi -= 2 * pi * np.floor(phi / (2 * pi))
    return phi
