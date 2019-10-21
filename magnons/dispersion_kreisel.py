import matplotlib.pyplot as plt
import numpy as np
from magnons.yig import a, S, mu, J, Ms
from magnons.cgs import E_to_GHz
from magnons.energies import energies, energies_uniform_mode, get_dispersion_theta
from tqdm import tqdm
from copy import copy
from numpy import pi

if __name__ == "__main__":
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
        "Nr": 4,
        "Ng": 4,
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

    for theta in tqdm([0, 30, 60, 90]):
        plt.figure()
        res, kvalues = get_dispersion_theta(np.radians(theta),
                                            30,
                                            firstN=6,
                                            parallel=True,
                                            **kwargs)
        absk = np.sqrt(np.sum(kvalues**2, axis=1))
        for i in range(6):
            plt.semilogx(absk, res[:, i], "-", color="black")

        uni = [energies_uniform_mode(*x, **kwargs_uni) for x in kvalues]
        plt.semilogx(absk, uni, "r--")
        plt.axhline(E0)
        minimum = res.min()
        plt.ylim(top=5.5, bottom=minimum * 0.95)
        plt.xlabel(r"$k\ (cm^{-1})$")
        plt.ylabel(r"$E_k\ (GHz)$")

        plt.savefig(f"./Kreisel_plots/theta_{theta}_withtranspose.pdf")
    plt.show()
